import argparse
import os
import time
from utils_common import get_free_node
def ligpre(csv,mae,nchir,ligpre_path='/public/software/schrodinger/2020-3/ligprep',device='gpu'):

    with open('utils/ligpre/simi.inp','w') as inp:
        inp.write(f'INPUT_FILE_NAME   {csv}\n'
                    f'OUT_MAE   {mae}\n'
                    'FORCE_FIELD   16\n'
                    'EPIK   yes\n'
                    'DETERMINE_CHIRALITIES   no\n'
                    'IGNORE_CHIRALITIES   no\n'
                    f'NUM_STEREOISOMERS   {nchir}\n'
                    )
    node,partion,ncpu=get_free_node(device=device)

    # We run glide on another machine. If you don't run it this way, please modify the following code!!
    njobs=ncpu*4
    with open('utils/ligpre/simi.sh','w') as inp:
        inp.write('#!/bin/bash\n'
                    '#SBATCH --nodes=1\n'
                    f'#SBATCH -w {node}\n'
                    f'#SBATCH --partition={partion}\n'
                    f'#SBATCH --ntasks-per-node={ncpu}\n'
                    '#SBATCH --time=100000:00:00\n'
                    '#SBATCH --job-name=ligpre\n'
                    f'{ligpre_dir} -inp simi.inp -HOST localhost:{ncpu} -NJOBS {njobs} -TMPLAUNCHDIR -WAIT\n'
                    )
    os.system('ssh 192.168.2.103 \'cd /public/home/chensheng/project/aixfuse2/utils/ligpre && sbatch simi.sh\'')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--mae')
    parser.add_argument('--nchir',default=32)
    parser.add_argument('--njobs',default=368)

    args = parser.parse_args()
    ligpre(args.csv,args.mae,args.nchir)