from pathlib import Path
import os,sys
# Fix head-side BLAS thread count for repeatable timing and to avoid nested parallelism.
for env in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[env]=os.environ.get('LIRA_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from lira_rs.cli import main
if __name__=='__main__': main()
