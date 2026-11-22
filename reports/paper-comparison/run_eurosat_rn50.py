from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import csv
import io
import os
import random
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from lira_rs.core import Config, fit, score
from lira_rs.data import filehash, load_json, perturb, save_json
from lira_rs.evaluation import metrics
from lira_rs.native import FILES, RN_SHA
from lira_rs.pipeline import put_npz


METHODS = ('visual', 'timo-proto', 'timo-loo', 'lira')
SHOTS = (1, 2, 4, 8, 16)
SEEDS = (1, 2, 3)


def main():
    sources = load_json(ROOT / 'reports/sources.json')['resources']
    for key in ('EuroSAT RGB', 'EuroSAT CoOp split', 'OpenAI RN50', 'TIMO'):
        entry = sources[key]
        if filehash(ROOT / entry['path']) != entry['sha256']:
            raise ValueError(f'Fingerprint mismatch for {key}')
    upstream = ROOT / 'data/upstream'
    for path in ('clip/__init__.py', 'clip/clip.py', 'clip/model.py', 'clip/simple_tokenizer.py', 'clip/bpe_simple_vocab_16e6.txt.gz'):
        with ZipFile(upstream / 'TIMO.zip') as author:
            local = upstream / 'local-timo' / path
            if local.read_bytes() != author.read(f'{FILES["timo"][1]}/{path}'):
                raise ValueError(f'Author CLIP archive mismatch for {path}')
    sys.path.insert(0, str(upstream / 'local-timo'))
    import clip
    from clip.clip import _transform
    from clip.model import build_model

    weight = ROOT / 'data/weights/RN50.pt'
    if filehash(weight) != RN_SHA:
        raise ValueError('RN50 checksum mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('This RN50 experiment uses the CUDA author encoder')
    with weight.open('rb') as stream:
        checkpoint = torch.jit.load(stream, map_location='cpu').state_dict()
    model = build_model(checkpoint).cuda().eval().requires_grad_(False)
    transform = _transform(model.visual.input_resolution)
    split_path = upstream / 'coop-split-usable.json'
    split = load_json(split_path)
    train = defaultdict(list)
    classes = [None] * 10
    for row in split['train']:
        train[row[1]].append(row)
        classes[row[1]] = Path(row[0]).parts[0].lower()
    if [len(split[k]) for k in ('train', 'val', 'test')] != [13500, 5400, 8100]:
        raise ValueError('Unexpected CoOp split sizes')
    episodes = {}
    for k in SHOTS:
        for seed in SEEDS:
            rng = random.Random(seed)
            episodes[k, seed] = [row for category in train.values() for row in rng.sample(category, k)]
    support_pool = sorted(set(row[0] for episode in episodes.values() for row in episode))
    query_ids = [row[0] for row in split['test']]
    if len(set(query_ids)) != 8100 or set(query_ids) & set(support_pool):
        raise ValueError('Image names are repeated or overlap support and query')
    dictionary = load_json(ROOT / 'prompts/eurosat.json')
    prompts, text_meta = perturb(dictionary['classes'], classes, 'normal', 7701)

    @torch.inference_mode()
    def encode_images(archive, names):
        features = []
        for start in range(0, len(names), 64):
            batch = []
            for name in names[start:start + 64]:
                with Image.open(io.BytesIO(archive.read(f'2750/{name}'))) as image:
                    batch.append(transform(image.convert('RGB')))
            encoded = model.encode_image(torch.stack(batch).cuda()).float()
            features.append(torch.nn.functional.normalize(encoded, dim=1).cpu().numpy())
        return np.concatenate(features)

    with ZipFile(ROOT / sources['EuroSAT RGB']['path']) as data:
        z = encode_images(data, support_pool + query_ids)
    with torch.inference_mode():
        tokens = clip.tokenize([prompt for category in prompts for prompt in category], truncate=True).cuda()
        t = model.encode_text(tokens).float().cpu().numpy().reshape(10, 4, -1)
    support_index = {name: i for i, name in enumerate(support_pool)}
    query_z = z[len(support_pool):]
    cfg = load_json_or_yaml(ROOT / 'configs/main.yaml')
    cfg['lambdas'] = tuple(cfg['lambdas'])
    config = Config(**cfg)

    fit_paths = []
    for (k, seed), episode in episodes.items():
        ids = [row[0] for row in episode]
        indices = [support_index[name] for name in ids]
        labels = np.array([row[1] for row in episode], dtype=np.int64)
        if not np.all(np.bincount(labels, minlength=10) == k):
            raise ValueError('Support budget differs by class')
        run = OUT / 'eurosat-rn50' / f'K{k:02}' / f'seed-{seed}'
        save_json(run / 'episode.json', {'ids': ids, 'labels': labels.tolist(), 'shot': k, 'seed': seed})
        for method in METHODS:
            state = fit(z[indices], labels, t, config, ids=ids, method=method)
            out = run / method
            pred_file = out / 'predictions.npz'
            put_npz(pred_file, ids=query_ids, logits=score(query_z, state).astype(np.float32))
            fit_paths.append((k, seed, method, pred_file, state.log.get('lambda'), state.log.get('fallback_reason')))
        print(f'EuroSAT RN50 K={k} seed={seed} scored', flush=True)

    # Query truth is used only after fitting and saving predictions for all settings.
    y = np.array([row[1] for row in split['test']], dtype=np.int64)
    results = []
    for k, seed, method, pred_file, strength, fallback in fit_paths:
        with np.load(pred_file, allow_pickle=False) as data, np.load(pred_file.parents[1] / 'visual/predictions.npz', allow_pickle=False) as visual:
            if not np.array_equal(data['ids'], query_ids) or not np.array_equal(visual['ids'], query_ids):
                raise ValueError('Query IDs were reordered')
            logits = data['logits']
            summary = metrics(logits, y, ids=query_ids, base=visual['logits'])
        correct = int(np.count_nonzero(logits.argmax(1) == y))
        results.append({'dataset': 'EuroSAT RGB', 'backbone': 'RN50', 'K': k, 'seed': seed, 'method': method,
                        'support_labels': 10 * k, 'validation_labels': 0, 'query_n': len(y), 'correct': correct,
                        **{key: summary[key] for key in ('accuracy', 'macro_f1', 'brier', 'nll', 'ece', 'aurc', 'harm', 'rescue')},
                        'lambda': strength, 'fallback': fallback, 'predictions_sha256': filehash(pred_file)})
    with (OUT / 'eurosat_rn50_runs.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    save_json(OUT / 'eurosat-rn50-provenance.json', {
        'split_sha256': filehash(split_path), 'archive_sha256': sources['EuroSAT RGB']['sha256'],
        'rn50_sha256': RN_SHA, 'timo_archive_sha256': sources['TIMO']['sha256'], 'prompt_hash': text_meta['prompt_hash'],
        'view': 'one deterministic center view per image', 'sampling': 'Python random.sample per class in file order, seeded independently per K',
        'deduplication': 'original CoOp split without additional project-level deduplication',
        'query_count': len(y), 'validation_labels_used': 0, 'support_pool_count': len(support_pool),
        'python': sys.version, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0), 'runs': len(results),
    })
    print(f'Saved {len(results)} runs with {len(y)} queries each', flush=True)


def load_json_or_yaml(path):
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8'))['head']


if __name__ == '__main__':
    main()
