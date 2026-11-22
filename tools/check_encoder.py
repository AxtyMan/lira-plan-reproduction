"""Re-encode original RGB images and prompts, verify the frozen feature cache, and time a cold episode."""
from pathlib import Path
from time import perf_counter
import ctypes
from ctypes import wintypes
import io
import os
import sys
from zipfile import ZipFile

START = perf_counter()
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lira_rs import core, data, pipeline
from lira_rs.cli import config, read_images, read_text
from lira_rs.features import Encoder


def peak_working_set():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    read_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    read_memory.restype = wintypes.BOOL
    ok = read_memory(process(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError()
    return counters.peak_ws


def main():
    settings, head = config(ROOT/'configs/main.yaml')
    encoder = Encoder(ROOT/'data/weights/clip', 'cuda')
    output = ROOT/'reports'/'encoder-check.json'
    report = {'backbone':encoder.meta,'blas_threads':2,'warmup_batches':0,'datasets':{}}
    for dataset in ('ucmerced','eurosat','resisc45'):
        start = perf_counter()
        manifest_path, manifest, cache, _, meta = read_images(dataset)
        assert data.filehash(manifest['archive']) == meta['archive_sha256'] == manifest['archive_sha256']
        rows = manifest['images']
        assert np.array_equal(cache['ids'], [row['id'] for row in rows])
        np.testing.assert_array_equal(cache['projection'],encoder.model.visual_projection.weight.detach().cpu().numpy().T)
        batch = meta['batch']
        maximum = np.zeros(2)
        torch.cuda.reset_peak_memory_stats()
        # Query labels remain sealed during encoding and subsequent candidate selection.
        encoded = np.empty_like(cache['z'])
        with ZipFile(manifest['archive']) as archive:
            for left in range(0,len(rows),batch):
                selected = rows[left:left+batch]
                images = []
                for row in selected:
                    with Image.open(io.BytesIO(archive.read(row['member']))) as im:
                        images.append(im.convert('RGB'))
                z, pre = encoder.images(images)
                end = left + len(z)
                errors = (np.max(np.abs(z-cache['z'][left:end])),np.max(np.abs(pre-cache['pen'][left:end])))
                maximum = np.maximum(maximum,errors)
                np.testing.assert_allclose(z,cache['z'][left:end],rtol=1e-5,atol=2e-6)
                np.testing.assert_allclose(pre,cache['pen'][left:end],rtol=1e-5,atol=2e-5)
                encoded[left:end] = z
        image_sec = perf_counter()-start
        text_max = 0.
        configs = [(c,7701) for c in settings['conditions']] + [('wrong30',s) for s in settings['text_seeds'][1:]]
        for condition,seed in configs:
            old, txmeta = read_text(dataset,condition,seed,meta)
            current, mask = encoder.texts(txmeta['prompts'])
            assert np.array_equal(mask, old['mask'])
            np.testing.assert_allclose(current,old['t'],rtol=1e-5,atol=2e-6)
            text_max = max(text_max, float(np.max(np.abs(current-old['t']))))
            if condition == 'normal':
                normal = {**old,'t':current,'mask':mask}
        cell = ROOT/'results'/'from-rgb'/dataset
        pipeline.frozen_episode(manifest_path,4,101,'A',cell/'episode.json')
        episode = data.load_json(cell/'episode.json')
        idx = {row['id']:i for i,row in enumerate(rows)}
        features = {**cache,'z':encoded}
        for method in ('visual','lira'):
            st, m = pipeline.fit_artifact(episode,features,normal,method,head,cell/method,idx)
            pipeline.score_artifact(manifest,features,idx,st,m,cell/method)
            result = pipeline.evaluate_artifact(manifest_path,cell/method,cell/'visual')
        report['datasets'][dataset] = dict(images=len(rows),batch=batch,image_seconds=image_sec,text_conditions=len(configs),
            max_abs_z=float(maximum[0]), max_abs_pre=float(maximum[1]), max_abs_text=text_max,
            lira_k4_s101_accuracy=result['accuracy'],gpu_peak_allocated=torch.cuda.max_memory_allocated(),
            process_peak_ws_bytes=peak_working_set(),total_dataset_seconds=perf_counter()-start,
            elapsed_from_process_start_seconds=perf_counter()-START,cache_bytes=meta['bytes'])
        print(dataset,report['datasets'][dataset],flush=True)
        data.save_json(output,report)


if __name__ == '__main__':
    main()
