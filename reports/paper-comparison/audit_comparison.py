"""Verify source fingerprints and recalculate paper comparison tables from per-image scores."""
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
import re
from statistics import mean, stdev
import sys
from zipfile import ZipFile

import numpy as np
from scipy.special import softmax

from run_eurosat_rn50 import METHODS, OUT, ROOT, SEEDS, SHOTS
from lira_rs.data import filehash, load_json, save_json


PERCENT_KEYS = ('accuracy', 'macro_f1', 'harm', 'rescue')
METRIC_KEYS = (*PERCENT_KEYS, 'brier', 'nll', 'ece', 'aurc')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(logits, y, ids, visual):
    """Keep an independent metric calculation for audit of the normal evaluator."""
    p = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    predicted = logits.argmax(axis=1)
    ok = predicted == y
    v = visual.argmax(axis=1) == y
    conf = p.max(axis=1)
    bins = np.minimum(np.floor(conf * 15).astype(int), 14)
    error = abs(np.bincount(bins, weights=ok.astype(float) - conf, minlength=15)).sum() / len(y)
    order = np.lexsort((ids, -conf))
    correct = int(ok.sum())
    classes = p.shape[1]
    matching = np.bincount(y[ok], minlength=classes)
    totals = np.bincount(y, minlength=classes) + np.bincount(predicted, minlength=classes)
    f1 = np.divide(2 * matching, totals, out=np.zeros(classes), where=totals > 0).mean()
    return {
        'correct': correct,
        'accuracy': correct / len(y),
        'macro_f1': f1,
        'harm': np.mean(~ok & v),
        'rescue': np.mean(ok & ~v),
        'brier': np.square(p - np.eye(p.shape[1])[y]).sum(axis=1).mean(),
        'nll': -np.log(p[np.arange(len(y)), y].clip(1e-12, 1)).mean(),
        'ece': error,
        'aurc': (np.cumsum(~ok[order]) / np.arange(1, len(y)+1)).mean(),
    }


def verify_numbers(actual, recorded, context):
    for key in METRIC_KEYS:
        if not np.isclose(actual[key], float(recorded[key]), rtol=0, atol=2e-12):
            raise ValueError(f'{context}: {key} mismatch: {actual[key]} != {recorded[key]}')


def audit_sources():
    source = load_json(ROOT / 'reports/sources.json')
    for paper in source['provided_papers'].values():
        assert filehash(ROOT.parent / paper['file']) == paper['sha256']
    timo = source['resources']['TIMO']
    assert filehash(ROOT / timo['path']) == timo['sha256']
    split = source['resources']['EuroSAT CoOp split']
    assert filehash(ROOT / split['path']) == split['sha256']
    return timo, load_json(ROOT / split['path'])


def official_rows(source, query_n):
    rows = []
    means = {}
    # Retain both the PDF values and the repository averages for eleven datasets.
    table_checks = []
    table1 = {'TIMO': (66.43, 68.15, 71.20, 73.55, 76.61),
              'TIMO_S': (66.78, 68.69, 71.63, 74.20, 76.86)}
    with ZipFile(ROOT / source['path']) as archive:
        for method, paper_means in table1.items():
            member = f'TIMO-main/outputs/{method}.txt'
            output = archive.read(member).decode('utf-8')
            values = {(d, int(k), int(s)): float(v) for d, k, s, v in re.findall(
                r'^([a-z0-9_]+)_(\d+)_(\d+): ([\d.]+)\s*$', output, re.M)}
            datasets = {key[0] for key in values}
            assert len(datasets) == 11 and len(values) == 165
            for k, paper_mean in zip(SHOTS, paper_means):
                means[k, method] = [values['eurosat', k, s] for s in SEEDS]
                repository_mean = mean(values[d, k, s] for d, s in product(datasets, SEEDS))
                if method == 'TIMO':
                    assert round(repository_mean, 2) == paper_mean
                table_checks.append(dict(method=method, K=k, pdf_top1_pct=paper_mean,
                                         repository_top1_pct=repository_mean, difference_pp=repository_mean - paper_mean))
                for seed, value in zip(SEEDS, means[k, method]):
                    reconstructed_correct = round(query_n * value / 100)
                    assert abs(value - 100 * reconstructed_correct / query_n) < 1e-10
                    rows.append(dict(dataset='EuroSAT RGB', backbone='RN50', K=k, seed=seed, method=method,
                                     top1_pct=value, query_n_local_coop=query_n, correct_implied_by_rate=reconstructed_correct,
                                     source_archive=source['path'], member=member, source_key=f'eurosat_{k}_{seed}'))
    write_csv(OUT / 'timo_author_eurosat.csv', rows)
    write_csv(OUT / 'timo_table1_crosscheck.csv', table_checks)
    return means


def audit_rn50(split):
    query_ids = np.array([row[0] for row in split['test']])
    truth = np.array([row[1] for row in split['test']])
    rows = read_csv(OUT / 'eurosat_rn50_runs.csv')
    indexed = {(int(r['K']), int(r['seed']), r['method']): r for r in rows}
    assert len(rows) == len(indexed) == len(SHOTS) * len(SEEDS) * len(METHODS)
    by_method = defaultdict(list)
    for k, seed in product(SHOTS, SEEDS):
        cell = OUT / 'eurosat-rn50' / f'K{k:02}' / f'seed-{seed}'
        with np.load(cell / 'visual/predictions.npz', allow_pickle=False) as file:
            assert np.array_equal(file['ids'], query_ids)
            visual = file['logits']
        for method in METHODS:
            row = indexed[k, seed, method]
            assert int(row['query_n']) == len(query_ids) == 8100
            assert int(row['support_labels']) == 10 * k and row['validation_labels'] == '0'
            pred_file = cell / method / 'predictions.npz'
            assert filehash(pred_file) == row['predictions_sha256']
            with np.load(pred_file, allow_pickle=False) as file:
                assert set(file.files) == {'ids', 'logits'} and np.array_equal(file['ids'], query_ids)
                actual = measure(file['logits'], truth, query_ids, visual)
            verify_numbers(actual, row, pred_file)
            assert actual['correct'] == int(row['correct'])
            by_method[k, method].append(actual['accuracy'] * 100)
    return by_method


def audit_project_main():
    source = ROOT / 'results/replay-primary/cells/resisc45/A/K04'
    summary = ROOT / 'results/summary.csv'
    by_method = defaultdict(list)
    conditions = ('normal', 'wrong30')
    methods = ('visual', 'timo-proto', 'lira', 'timo-loo', 'prolip')
    support_seeds = (101, 103, 107, 109, 113, 127, 131, 137, 139, 149)
    sealed = {r['id']: r['label'] for r in load_json(ROOT / 'data/manifests/resisc45/query.labels.json')}
    for seed, condition in product(support_seeds, conditions):
        cell = source / f'S{seed}' / f'{condition}-7701'
        with np.load(cell / 'visual/evaluated.npz', allow_pickle=False) as v:
            ids, y, visual = v['ids'], v['labels'], v['logits']
        assert len(y) == len(set(ids)) == len(sealed) == 12598
        assert np.array_equal(y, [sealed[name] for name in ids])
        for method in methods:
            runs = [cell / method]
            if method == 'prolip':
                runs = [cell / method / f'optimization-{opt}' for opt in (191, 193, 197)]
            seed_values = []
            for run in runs:
                saved = load_json(run / 'metrics.json')
                assert saved['meta']['role_sizes'] == {'support': 180}
                with np.load(run / 'evaluated.npz', allow_pickle=False) as file:
                    assert np.array_equal(file['ids'], ids) and np.array_equal(file['labels'], y)
                    actual = measure(file['logits'], y, ids, visual)
                    if method == 'lira':
                        assert saved['meta']['lambda'] == 0 and np.array_equal(file['logits'], visual)
                verify_numbers(actual, saved['raw'], run)
                seed_values.append(actual)
            by_method[condition, method].append({k: mean(r[k] for r in seed_values) for k in METRIC_KEYS})
    official_summary = {(r['condition'], r['requested_method']): r for r in read_csv(summary)
                        if r['dataset']=='resisc45' and r['protocol']=='A' and r['K']=='4' and r['text_seed']=='7701' and r['variant']=='main'}
    rows = []
    for (condition, method), values in by_method.items():
        result = {k: mean(r[k] for r in values) for k in METRIC_KEYS}
        verify_numbers(result, official_summary[condition, method], f'project summary {condition}/{method}')
        rows.append(dict(condition=condition, method=method, query_n=12598, seed_count=len(support_seeds),
                         **result, accuracy_sd=stdev(r['accuracy'] for r in values)))
    write_csv(OUT / 'resisc45_verified_metrics.csv', rows)
    return rows


def main():
    timo, split = audit_sources()
    author = official_rows(timo, len(split['test']))
    local = audit_rn50(split)
    rows = []
    for k in SHOTS:
        lira = mean(local[k, 'lira'])
        row = {'K': k, 'query_n': len(split['test'])}
        for source, methods in ((author, ('TIMO', 'TIMO_S')), (local, METHODS)):
            for name in methods:
                key = name.replace('-', '_').lower()
                row[f'{key}_top1_pct'] = mean(source[k, name])
                row[f'{key}_top1_sd_pp'] = stdev(source[k, name])
        row['lira_minus_author_timo_pp'] = lira - mean(author[k, 'TIMO'])
        row['lira_minus_same_budget_timo_loo_pp'] = lira - mean(local[k, 'timo-loo'])
        rows.append(row)
    write_csv(OUT / 'eurosat_top1_comparison.csv', rows)

    nearby = []
    query_ids = np.array([row[0] for row in split['test']])
    truth = np.array([row[1] for row in split['test']])
    for method in ('timo', 'prolip'):
        base = ROOT / f'results/native/{method}-eurosat-rn50-k4'
        native = load_json(base / 'native-report.json')
        with np.load(base / 'evaluated.npz', allow_pickle=False) as file:
            assert np.array_equal(file['ids'], query_ids) and np.array_equal(file['labels'], truth)
            actual = measure(file['logits'], truth, query_ids, file['logits'])
        for key in ('accuracy', 'macro_f1', 'brier', 'nll', 'ece'):
            assert np.isclose(actual[key], native['metrics'][key], rtol=0, atol=2e-12)
        nearby.append(dict(source='local author-style replay', method=method, K=4, seed=1, correct=actual['correct'],
                           query_n=len(truth), top1_pct=100*actual['accuracy'],
                           validation_labels=native['source_details']['validation_labels_used']))
    write_csv(OUT / 'native_replay_top1.csv', nearby)
    project = audit_project_main()
    save_json(OUT / 'verification.json', {
        'paper_pdf_sha256_verified': 3, 'author_eurosat_runs': 30,
        'matched_project_eurosat_runs_recomputed': 60, 'author_style_local_replays_recomputed': len(nearby),
        'resisc45_metrics_recomputed': 140, 'resisc45_aggregated_rows': len(project),
        'verified_metrics': list(METRIC_KEYS), 'resisc45_query_count': 12598, 'eurosat_query_count': 8100,
    })
    for row in rows:
        print(row['K'], *(f'{row[key]:.5f}' for key in ('timo_top1_pct', 'timo_s_top1_pct', 'visual_top1_pct', 'timo_proto_top1_pct', 'timo_loo_top1_pct', 'lira_top1_pct', 'lira_minus_author_timo_pp', 'lira_minus_same_budget_timo_loo_pp')))
    print('Verified 3 PDF fingerprints, 30 author rows, 60 RN50 runs, 2 local native replays and 140 RESISC45 runs.')


if __name__ == '__main__':
    main()
