import json
from pathlib import Path
D=Path(__file__).resolve().parent
raw={}
for line in (D/'definitions.psv').read_text(encoding='utf-8-sig').splitlines():
 k,name,*a,other=line.split('|'); raw[k.lower()]={'core':f'An aerial image of {name}.','attrs':[f'{x[0].upper()}{x[1:]}.' for x in a], 'paraphrase':[f'From overhead, {x}.' for x in a],'confuser':other}
uc='agricultural airplane baseballdiamond beach buildings chaparral denseresidential forest freeway golfcourse harbor intersection mediumresidential mobilehomepark overpass parkinglot river runway sparseresidential storagetanks tenniscourt'.split()
eu='annualcrop forest herbaceousvegetation highway industrial pasture permanentcrop residential river sealake'.split()
nw='airplane airport baseball_diamond basketball_court beach bridge chaparral church circular_farmland cloud commercial_area dense_residential desert forest freeway golf_course ground_track_field harbor industrial_area intersection island lake meadow medium_residential mobile_home_park mountain overpass palace parking_lot railway railway_station rectangular_farmland river roundabout runway sea_ice ship snowberg sparse_residential stadium storage_tank tennis_court terrace thermal_power_station wetland'.split()
alias=dict(annualcrop='agriculture',baseballdiamond='baseball_diamond',denseresidential='dense_residential',freeway='highway',golfcourse='golf_course',mediumresidential='medium_residential',mobilehomepark='mobile_home_park',parkinglot='parking_lot',sparseresidential='sparse_residential',storagetanks='storage_tank',tenniscourt='tennis_court')
fix={'eurosat':{'forest':'herbaceousvegetation','river':'sealake'},'ucmerced':{'baseballdiamond':'golfcourse','golfcourse':'agricultural','tenniscourt':'baseballdiamond','storagetanks':'buildings','river':'harbor'},'resisc45':{'freeway':'railway'}}
sources={'eurosat':'Helber et al., DOI 10.1109/JSTARS.2019.2918242; phelber/EuroSAT RGB class taxonomy','ucmerced':'Yang and Newsam, DOI 10.1145/1869790.1869829; TorchGeo v0.7.1 class taxonomy','resisc45':'Cheng et al., DOI 10.1109/JPROC.2017.2675998; TorchGeo v0.7.1 class taxonomy'}
for dataset, classes in zip(sources,[eu,uc,nw]):
 reverse={alias.get(c,c):c for c in classes}; out={}
 for c in classes:
  v=raw[alias.get(c,c)].copy(); other=fix[dataset].get(c,reverse.get(v['confuser'],v['confuser']))
  assert other in classes and other!=c,(dataset,c,other)
  v.update(confuser=other); out[c]=v
 (D/f'{dataset}.json').write_text(json.dumps({'version':'taxonomy-v1','source':sources[dataset],'notes':'Independently authored static visible attributes; not image-level ground truth. Paraphrase uses an alternate overhead framing. No query outcomes used.', 'classes':out},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
