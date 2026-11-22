"""Frozen official CLIP ViT-B/32 converted weights; local-only inference."""
import io
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel
from .data import digest, filehash, load_json, save_json

OFFICIAL_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CONVERTED_SHA='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'


def clean_assert_weights(model,official):
    if filehash(official)!=OFFICIAL_SHA: raise ValueError('OpenAI CLIP model JIT archive SHA-256 mismatch')
    with Path(official).open('rb') as f: src=torch.jit.load(f,map_location='cpu').state_dict()
    dst=model.state_dict()
    names={'visual.conv1.weight':'vision_model.embeddings.patch_embedding.weight',
           'visual.class_embedding':'vision_model.embeddings.class_embedding',
           'visual.positional_embedding':'vision_model.embeddings.position_embedding.weight',
           'token_embedding.weight':'text_model.embeddings.token_embedding.weight',
           'positional_embedding':'text_model.embeddings.position_embedding.weight','logit_scale':'logit_scale'}
    for a,b in [('visual.ln_pre','vision_model.pre_layrnorm'),('visual.ln_post','vision_model.post_layernorm'),('ln_final','text_model.final_layer_norm')]:
        for suffix in ['weight','bias']: names[a+'.'+suffix]=b+'.'+suffix
    for a,b in names.items():
        if not torch.equal(src[a].float(),dst[b].float()): raise ValueError(f'CLIP tensor mismatch: {a}')
    checked=set(names.values())
    for a,b in [('visual.proj','visual_projection.weight'),('text_projection','text_projection.weight')]:
        if not torch.equal(src[a].T.float(),dst[b].float()): raise ValueError(f'CLIP projection mismatch: {a}')
        checked.add(b)
    for pref,out in [('visual.transformer.resblocks','vision_model.encoder.layers'),('transformer.resblocks','text_model.encoder.layers')]:
        for i in range(12):
            a=f'{pref}.{i}.'; b=f'{out}.{i}.'
            for k,l in [('ln_1','layer_norm1'),('ln_2','layer_norm2'),('mlp.c_fc','mlp.fc1'),('mlp.c_proj','mlp.fc2'),('attn.out_proj','self_attn.out_proj')]:
                for s in ('weight','bias'):
                    if not torch.equal(src[a+k+'.'+s].float(),dst[b+l+'.'+s].float()): raise ValueError(f'CLIP tensor mismatch: {a+k}')
                    checked.add(b+l+'.'+s)
            for s in ('weight','bias'):
                for k,v in zip(('q','k','v'),src[a+'attn.in_proj_'+s].chunk(3)):
                    n=b+'self_attn.'+k+'_proj.'+s
                    if not torch.equal(v.float(),dst[n].float()): raise ValueError(f'CLIP tensor mismatch: {n}')
                    checked.add(n)
    if set(dst)-checked: raise ValueError(f'unchecked tensors {set(dst)-checked}')
    return len(checked)


class Encoder:
    def __init__(self,weights,device='auto'):
        start=perf_counter(); w=Path(weights)
        wanted='cuda' if torch.cuda.is_available() else 'cpu'
        self.device=torch.device(wanted if device=='auto' else device)
        if self.device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        f=w/'pytorch_model.bin'; official=w.parent/'ViT-B-32.pt'; self.weight_sha=filehash(f)
        if self.weight_sha!=CONVERTED_SHA: raise ValueError('fixed CLIP conversion SHA-256 mismatch')
        # HF uses PyTorch >=2.6 safe weights-only loading. All tensors compared against fixed OpenAI checkpoint.
        self.model=CLIPModel.from_pretrained(w,local_files_only=True,use_safetensors=False,weights_only=True).eval().requires_grad_(False)
        n=clean_assert_weights(self.model,official)
        self.model.to(self.device)
        self.processor=CLIPImageProcessor.from_pretrained(w,local_files_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(w,local_files_only=True)
        self.meta=dict(architecture='CLIP ViT-B/32',revision=REVISION,weight_sha256=self.weight_sha,official_sha256=OFFICIAL_SHA,verified_tensors=n,
                       device=str(self.device),device_name=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'CPU',torch=torch.__version__,
                       preprocessing=self.processor.to_dict(),tokenizer_sha256=filehash(w/'tokenizer.json'),load_seconds=perf_counter()-start)

    @torch.inference_mode()
    def images(self,imgs):
        x=self.processor(images=imgs,return_tensors='pt').pixel_values.to(self.device)
        pen=self.model.vision_model(pixel_values=x).pooler_output
        return torch.nn.functional.normalize(self.model.visual_projection(pen)).cpu().numpy(),pen.cpu().numpy()

    @torch.inference_mode()
    def texts(self,rows):
        mask=np.array([[p is not None for p in r] for r in rows]); flat=[p for r in rows for p in r if p is not None]
        x=self.tokenizer(flat,padding=True,truncation=True,max_length=77,return_tensors='pt').to(self.device)
        y=self.model.text_model(**x).pooler_output; z=torch.nn.functional.normalize(self.model.text_projection(y)).cpu().numpy()
        values=np.zeros((*mask.shape,z.shape[-1]),np.float32); values[mask]=z
        return values,mask


def atomic_npz(path,**values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); t=path.with_name(path.name+'.tmp')
    with t.open('wb') as f: np.savez_compressed(f,**values)
    t.replace(path)


def extract_image(encoder,manifest,out,batch=64,*,native_validation=False):
    manifest=Path(manifest); m=load_json(manifest); out=Path(out); archive=Path(m['archive'])
    if filehash(archive)!=m['archive_sha256']: raise ValueError('dataset archive changed after split freeze')
    scans=m['images'] if not native_validation else load_json(manifest.with_name('native-val.images.json'))
    meta={**encoder.meta,'split_hash':m['split_hash'],'dataset':m['dataset'],'view':'deterministic_center','batch':batch,'image_count':len(scans),'archive_sha256':m['archive_sha256']}
    key=digest({k:v for k,v in meta.items() if k!='load_seconds'})
    start=perf_counter(); allz=[]; allpen=[]
    if encoder.device.type=='cuda': torch.cuda.reset_peak_memory_stats(encoder.device)
    with ZipFile(archive) as z:
        for j in range(0,len(scans),batch):
            ims=[]
            for r in scans[j:j+batch]:
                with Image.open(io.BytesIO(z.read(r['member']))) as im: ims.append(im.convert('RGB'))
            f,p=encoder.images(ims); allz.append(f); allpen.append(p)
            if j%(batch*20)==0 or j+batch>=len(scans): print(f"{m['dataset']} {min(j+batch,len(scans))}/{len(scans)}",flush=True)
    atomic_npz(out,ids=np.array([r['id'] for r in scans]),z=np.concatenate(allz),pen=np.concatenate(allpen),projection=encoder.model.visual_projection.weight.detach().cpu().numpy().T,cache_key=np.array(key),split_hash=np.array(m['split_hash']))
    meta.update(cache_key=key,seconds=perf_counter()-start,bytes=out.stat().st_size,cuda_peak_allocated=torch.cuda.max_memory_allocated(encoder.device) if encoder.device.type=='cuda' else None)
    save_json(out.with_suffix('.json'),meta)
    return meta
