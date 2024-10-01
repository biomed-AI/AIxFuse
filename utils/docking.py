import argparse
import os
import time
from utils_common import get_free_node
def docking(input,grid,glide_path='/public/software/schrodinger/2020-3/glide',prec='SP',path='utils/docking',device='gpu'):
    while not os.path.exists(input):
        time.sleep(100)
    pdb_id=grid.split('/')[-1].split('.')[0].split('_')[0]
    dir=os.path.join(path,f'{pdb_id}_{prec}')
    if not os.path.exists(dir):
        os.mkdir(dir)
    if prec=='XP':
        conf='POSTDOCK_XP_DELE   0.5\nPRECISION   XP\nWRITE_RES_INTERACTION   True\nWRITE_XP_DESC   True\n'
    elif prec=='HTVS':
        conf='PRECISION   HTVS'
    elif prec=='SP':
        conf='NENHANCED_SAMPLING   2\nPOSES_PER_LIG   3\nPOSTDOCK_NPOSE   100\nPRECISION   SP'
    
    with open(os.path.join(dir,'docking.in'),'w') as inp:
        inp.write(f'GRIDFILE   {grid}\n'
                    f'LIGANDFILE   {input}\n{conf}'
                    )
    node,partion,ncpu=get_free_node(device=device)


    # We run glide on another machine. If you don't run it this way, please modify the following code!!
    with open(os.path.join(dir,'docking.sh'),'w') as inp:
        inp.write('#!/bin/bash\n'
                '#SBATCH --nodes=1\n'
                f'#SBATCH -w {node}\n'
                f'#SBATCH --partition={partion}\n'
                f'#SBATCH --ntasks-per-node={ncpu}\n'
                '#SBATCH --time=100000:00:00\n'
                f'#SBATCH --job-name=docking_{dir}\n'
                f'{glide_dir}  docking.in -OVERWRITE -adjust -HOST localhost:{ncpu} -TMPLAUNCHDIR -WAIT'
                    )
    os.system(f'ssh 192.168.2.103 \'cd {dir} && sbatch docking.sh\'')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--grid')
    parser.add_argument('--prec',default='SP')

    args = parser.parse_args()
    docking(args.input,args.grid,args.prec)