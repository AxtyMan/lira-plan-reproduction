"""Import completed experiment inputs into this project without modifying the source.

Large immutable arrays and diagnostic blobs use NTFS hard links on the same volume.
Editable metadata is copied. Fresh and historical runs stay in distinct directories.
"""
from pathlib import Path
import argparse
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ('main', 'eurosat-main', 'resisc45-primary', 'resisc45-other', 'resisc45-k16',
           'description-controls', 'other-diagnostics')


def import_batch(source, target):
    count = 0
    for directory, _, filenames in os.walk(source):
        parent = target / Path(directory).relative_to(source)
        parent.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            original, dest = Path(directory) / name, parent / name
            if dest.exists():
                if dest.stat().st_size != original.stat().st_size:
                    raise ValueError(f'Incomplete snapshot entry: {dest}')
                continue
            if original.suffix in {'.gz', '.npz'}:
                try:
                    os.link(original, dest)
                except OSError:
                    shutil.copy2(original, dest)
            else:
                shutil.copy2(original, dest)
            count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_project', type=Path)
    source = parser.parse_args().previous_project.resolve()
    for batch in BATCHES:
        path = source / 'results' / batch
        if not (path / 'run-status.json').is_file():
            raise ValueError(f'Missing completed source batch: {path}')
        print(batch, import_batch(path, ROOT / 'results' / batch), flush=True)
