"""Lab acceptance from real immutable outputs; no synthetic data enter these reports."""
from pathlib import Path
from collections import defaultdict
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lira_rs.data import load_json, save_json, filehash
from lira_rs.pipeline import load_npz, aggregate


def main():
    root=ROOT/'results'; reports=ROOT/'reports'; figs=reports/'figures'; figs.mkdir(exist_ok=True)
    rows=aggregate(root)
    tables=reports/'tables'; tables.mkdir(exist_ok=True)
    for name,predicate in {
        '01-main':lambda r:r['protocol']=='A' and r['condition']=='normal' and r['variant']=='main',
        '02-text-quality':lambda r:r['protocol']=='A' and r['condition']!='normal' and r['variant']=='main',
        '03-label-probability':lambda r:r['condition']=='normal' and r['variant']=='main' and r['K']==4,
        '04-ablation':lambda r:r['variant']!='main',
        '05-efficiency':lambda r:r['condition']=='normal' and r['variant']=='main',
    }.items():
        subset=[r for r in rows if predicate(r)]
        if subset:
            with (tables/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as h:
                w=csv.DictWriter(h,fieldnames=list(subset[0])); w.writeheader(); w.writerows(subset)
    full=[p for p in root.glob('*/cells/**/metrics.json')]; fresh=set(); maincells={}; by={}; k1=[]; ncal=0
    for p in full:
        q=load_json(p); m=q['meta']; r=q['raw'];
        if abs(r['accuracy']-r['visual_accuracy']-r['rescue']+r['harm'])>1e-12: raise AssertionError(p)
        if q['calibrated'] is not None:
            if q['calibrated']['accuracy']!=r['accuracy']: raise AssertionError(p)
            ncal+=1
        if m['requested_method']=='lira' and m['protocol']=='A' and m['K']==1:
            assert m['lambda']==0 and m['fallback_reason']=='one_shot' and r['harm']==r['rescue']==0
            k1.append(str(p.relative_to(root)))
        run=p.relative_to(root).parts[0]
        if (m['protocol']=='A' and m['variant_name']=='main' and m['text_seed']==7701 and m['condition'] in ('normal','wrong30')
                and (run.startswith('replay-') or run in {'main','eurosat-main','resisc45-primary','resisc45-other','resisc45-k16'})):
            c=(m['dataset'],m['K'],m['support_seed'],m['condition'],m['requested_method'],m.get('optimization_seed')); maincells.setdefault(c,p)
            if run.startswith('replay-'): fresh.add(c)
        if m['dataset']=='resisc45' and m['K']==4 and m['protocol']=='A' and m.get('text_seed')==7701 and run in {'resisc45-primary','description-controls'}:
            sig=(m['condition'],m['requested_method'],m['variant_name']); by.setdefault(sig,[]).append((p,q))
    sets=['eurosat','ucmerced','resisc45']; seeds=[101,103,107,109,113,127,131,137,139,149]; methods=['visual','zs','ensemble','fixed','tip','gda-loo','timo-loo','timo-proto','prolip','lira']
    expected={(d,k,s,c,m,op) for d in sets for k in (1,2,4,8,16) for s in seeds for c in ('normal','wrong30') for m in methods for op in ([191,193,197] if m=='prolip' else [None])}
    missing=sorted(expected-set(maincells),key=str)
    for key in ('timo-eurosat-rn50-k4','prolip-eurosat-rn50-k4'):
        d=root/'native'/key; a=load_json(d/'native-report.json'); assert a['query_n']==8100 and a['support_labels']==40 and np.isfinite(a['metrics']['accuracy'])
    def finish(fig,name):
        fig.tight_layout(); fig.savefig(figs/(name+'.png'),dpi=165,bbox_inches='tight'); plt.close(fig)
    f,ax=plt.subplots(figsize=(11,2.9)); ax.set_axis_off()
    stages=['Freeze CLIP\n& roles','Class means\n+ TIMO IGT','Clip movement\nby class margin','Image LOO ×\n4 prompt masks','Admit by Brier\nor visual fallback','Score first;\nevaluate last']
    for i,stage in enumerate(stages):
        ax.text(.08+.168*i,.5,stage,transform=ax.transAxes,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.65','fc':'#e9f2f6','ec':'#47758c'})
        if i<5: ax.annotate('',xy=(.188+.168*i,.5),xytext=(.139+.168*i,.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#47758c'})
    finish(f,'01-method')
    f,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    for d,ax in zip(sets,axes):
        for meth in ('lira','timo-proto','visual','prolip'):
            rr=sorted([r for r in rows if r['dataset']==d and r['protocol']=='A' and r['condition']=='normal' and r['requested_method']==meth and r['variant']=='main'],key=lambda x:x['K']);
            ax.plot([r['K'] for r in rr],[100*r['accuracy'] for r in rr],marker='o',label=meth)
        ax.set_title(d); ax.set_xticks([1,2,4,8,16]); ax.set_xlabel('Labeled images / class'); ax.grid(alpha=.25)
    axes[0].set_ylabel('Accuracy (%)'); axes[-1].legend(fontsize=8); finish(f,'02-budget')
    f,ax=plt.subplots(figsize=(6,3.8)); conds=['normal','wrong10','wrong30','wrong50']
    for meth in ('lira','timo-proto','visual'):
        yy=[np.mean([q['raw']['harm'] for _,q in by.get((c,meth,'main'),[])])*100 if by.get((c,meth,'main')) else np.nan for c in conds]; ax.plot([0,10,30,50],yy,marker='o',label=meth)
    ax.set(xlabel='Target wrong attribute slots (%)',ylabel='Harm (% of queries)',title='RESISC45 4-shot'); ax.legend(); ax.grid(alpha=.25); finish(f,'03-harm')
    f,axs=plt.subplots(1,2,figsize=(9,3.7))
    for c,ax in zip(('normal','wrong30'),axs):
        for _,q in by.get((c,'lira','main'),[]):
            a=q['meta']; r=np.array(a['risk']); ax.plot(a['lambda_grid'],r[0]-r[0,0],color='#2a6f9e',alpha=.38); k=a['lambda_grid'].index(a['lambda']); ax.scatter(a['lambda'],r[0,k]-r[0,0],color='#c14c1b',s=14)
        ax.axhline(0,color='black',lw=.8); ax.set(xlabel='Candidate lambda',ylabel='Full-prompt Brier change',title=c)
    finish(f,'04-risk')
    f,ax=plt.subplots(figsize=(7,3.8)); dd=[]; names=[]
    for c in ('normal','wrong30'):
        for meth,v in [('lira','main'),('timo-proto','main'),('fixed-bounded','ablation')]:
            a=[w for _,q in by.get((c,meth,v),[]) for w in q['meta'].get('displacement',[])];
            if a: dd.append(a); names.append(f'{c}\n{meth}')
    if dd: ax.boxplot(dd,tick_labels=names,showfliers=False); ax.tick_params(axis='x',labelsize=8)
    ax.set_ylabel('Prototype displacement (L2)'); finish(f,'05-displacement')
    cases=[]
    manifest=load_json(ROOT/'data/manifests/resisc45/images.json'); members={r['id']:r['member'] for r in manifest['images']}
    import io
    from PIL import Image
    from zipfile import ZipFile
    f,axs=plt.subplots(1,3,figsize=(10.5,3.8))
    f.suptitle('RESISC45 4-shot · 30% wrong descriptions · fixed bounded control', fontsize=12)
    for ax,title in zip(axs,('Avoided harm','Missed rescue','Protection failure')):
        ax.set_axis_off()
        ax.set_title(title+'\nNot observed',fontsize=10)
    runs=sorted(by.get(('wrong30','lira','main'),[]),key=lambda item:item[1]['meta']['support_seed'])
    with ZipFile(manifest['archive']) as archive:
        for lpath,l in runs:
            if len(cases)==3:
                break
            bpath=lpath.parent.parent/'fixed-bounded--ablation'/'evaluated.npz'
            if not bpath.exists():
                continue
            lira=load_npz(lpath.parent/'evaluated.npz')
            control=load_npz(bpath)
            if not all(np.array_equal(lira[key],control[key]) for key in ('ids','labels','visual_predicted')):
                raise ValueError(f'unpaired case-study inputs: {lpath}')
            masks={'Avoided harm':control['harm'] & ~lira['harm'],
                   'Missed rescue':control['rescue'] & ~lira['rescue'],
                   'Protection failure':lira['harm']}
            for ax,(tag,mask) in zip(axs,masks.items()):
                if any(case['kind']==tag for case in cases) or not np.any(mask):
                    continue
                index=np.flatnonzero(mask)[0]
                image_id=str(lira['ids'][index])
                names=manifest['classes']
                true=names[lira['labels'][index]]
                lira_class=names[lira['predicted'][index]]
                control_class=names[control['predicted'][index]]
                with Image.open(io.BytesIO(archive.read(members[image_id]))) as image:
                    ax.imshow(image.convert('RGB'))
                ax.set_title(f'{tag}\nTrue: {true}\nLIRA: {lira_class}\nFixed bounded: {control_class}',fontsize=9)
                cases.append(dict(kind=tag,image_id=image_id,support_seed=l['meta']['support_seed'],
                                  true=true,lira=lira_class,control='fixed-bounded',control_class=control_class))
    finish(f,'06-cases'); save_json(figs/'06-cases.json',cases)
    save_json(reports/'acceptance.json',dict(main_expected=len(expected),main_completed=len(maincells),main_fresh=len(fresh),main_missing=[list(q) for q in missing],metrics_files=len(full),one_shot_lira_checked=len(k1),calibrated_fits_checked=ncal,report_rows=len(rows)))
    print(f'{len(maincells)}/{len(expected)} primary metric files; {len(full)} total; {len(k1)} 1-shot, {ncal} B calibrations checked')
    if '--strict' in sys.argv and (missing or ('--fresh' in sys.argv and fresh != expected)): raise SystemExit(2)

if __name__=='__main__':main()
