"""Check reproducibility of summary metrics by pairing fresh and historical fit keys."""
import csv
from pathlib import Path
from collections import Counter
import json

ROOT=Path(__file__).resolve().parents[1]
KEYS=('dataset','K','seed','protocol','condition','text_seed','method','variant','opt_seed')
METRICS=('accuracy','harm','rescue')

def main():
    def load(kind):
        path=ROOT/'reports'/f'{kind}-metrics-audit.csv'
        rows={}
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                key=tuple(row[k] for k in KEYS)
                if key not in rows or row['batch']=='resisc45-k16':
                    rows[key]=row
        return rows
    historical=load('archive');fresh=load('fresh')
    if not len(fresh)==3600 or not fresh.keys()<=historical.keys():
        raise ValueError('Expected 3600 fully paired fresh matrix cells')
    by_method=Counter(); max_difference={k:0. for k in METRICS}; changes=[]
    for key,row in fresh.items():
        reference=historical[key];d={k:float(row[k])-float(reference[k]) for k in METRICS}
        if any(d.values()):
            by_method[row['method']]+=1
            changes.append({k:row[k] for k in KEYS}|d)
        for k in METRICS:max_difference[k]=max(max_difference[k],abs(d[k]))
    result=dict(paired_fits=len(fresh),same_top1_harm_rescue=len(fresh)-len(changes),changed_by_method=dict(by_method),
                max_absolute_fraction_difference=max_difference,changed_cells=changes)
    (ROOT/'reports'/'repeatability.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='changed_cells'})

if __name__=='__main__':main()
