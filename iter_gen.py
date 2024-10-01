import sys
import os
import pandas as pd
sys.path.append('utils')
sys.setrecursionlimit(3000)
import argparse
import warnings
import pickle
import numpy as np
import dual_MCTS
from ligpre import ligpre
from docking import docking
from train import train
from tqdm import trange
import time
warnings.filterwarnings('ignore')

task={'rorgt_dhodh':{'pdb_id1':'5NTP','pdb_id2':'6QU7','prec':'XP','device':'gpu','color':['#006400','#B22222']},'gsk3b_jnk3':{'pdb_id1':'6Y9S','pdb_id2':'4WHZ','prec':'SP','device':'cpu','color':['#87CEEB','#D2691E']}}

def iter_gen(args,iter):
    target1,target2=args.task.split('_')
    pdb_id1=task[args.task]['pdb_id1']
    pdb_id2=task[args.task]['pdb_id2']
    prec=task[args.task]['prec']
    device=task[args.task]['device']
    generated_dir=os.path.join(args.generated_dir,f'{target1}_{target2}')
    cache_dir=os.path.join(args.cache_dir,f'{target1}_{target2}')
    model_dir=os.path.join(args.model_dir,f'{target1}_{target2}')
    iter_csv=os.path.join(generated_dir,f'gen_iter_{iter}.csv')
    agent_dir=os.path.join(cache_dir,f'dual_MCTS_Agent_iter_{max(0,iter-1)}.pkl')
    target1_pkl=os.path.join(cache_dir,f'core_info_{target1}.pkl')
    target2_pkl=os.path.join(cache_dir,f'core_info_{target2}.pkl')
    model_ckpt=os.path.join(model_dir,f'gen_iter_{max(0,iter-1)}.pt')
    final_agent=os.path.join(cache_dir,f'dual_MCTS_Agent_iter_{iter}.pkl')
    last_csv=os.path.join(generated_dir,f'util_iter_{max(0,iter-1)}.csv')
    iter_mae=os.path.join(args.ligpre_dir,f'{target1}_{target2}_gen_iter_{iter}.maegz')
    docking_dir='utils/docking'
    dir_5ntp=os.path.join(docking_dir,f'{pdb_id1}_{prec}')
    mae_5ntp=os.path.join(dir_5ntp,'docking_pv.maegz')
    dir_6qu7=os.path.join(docking_dir,f'{pdb_id2}_{prec}')
    mae_6qu7=os.path.join(dir_6qu7,'docking_pv.maegz')
    grid_5ntp=os.path.join(args.grid_dir,f'{pdb_id1}.zip')
    grid_6qu7=os.path.join(args.grid_dir,f'{pdb_id2}.zip')
    docking_csv=os.path.join(generated_dir,f'gen_iter_{iter}_scores.csv')
    next_csv=os.path.join(generated_dir,f'util_iter_{iter}.csv')
    new_5ntp='/'.join(mae_5ntp.split('/')[:-1]+[f'gen_iter_{iter}.maegz'])
    new_6qu7='/'.join(mae_6qu7.split('/')[:-1]+[f'gen_iter_{iter}.maegz'])
    new_model=os.path.join(model_dir,f'gen_iter_{iter}.pt')

    if not args.single_process or args.generate:
        # 蒙特卡洛分子生成
        if not os.path.exists(generated_dir):
            os.makedirs(generated_dir)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # if not os.path.exists(iter_csv):
        if iter<=1:
                if os.path.exists(agent_dir):
                    with open(agent_dir,'rb') as r:
                        dual_MCTS_Agent=pickle.load(r)
                else:
                    with open(target1_pkl,'rb') as r:
                        core_info_rorgt=pickle.load(r)
                    with open(target2_pkl,'rb') as r:
                        core_info_dhodh=pickle.load(r)
                    dual_MCTS_Agent=dual_MCTS.DualMCTS(core_info_rorgt,core_info_dhodh)
        else:
            with open(agent_dir,'rb') as r:
                dual_MCTS_Agent=pickle.load(r)
        if iter==0:
            if not os.path.exists(iter_csv):
                df=dual_MCTS_Agent.init_explore(args.gen_num)
                df.to_csv(iter_csv,index=False)
                with open(final_agent,'wb') as w:
                    pickle.dump(dual_MCTS_Agent,w)
        else:
            df_last=pd.read_csv(last_csv)
            docking_result={}
            for title,docking1,docking2 in zip(df_last['Title'],df_last[f'{pdb_id1}_{prec}'],df_last[f'{pdb_id2}_{prec}']):
                docking_result[title]={f'{pdb_id1}_{prec}_neg':-docking1 if docking1 else 1,f'{pdb_id2}_{prec}_neg':-docking2 if docking2 else 1}
            # if not os.path.exists(iter_csv):
            final_gen=True if iter==5 else False
            dual_MCTS_Agent.iter_explore(args.gen_num,model_path=model_ckpt,docking_result=docking_result,gen_csv=iter_csv,final_gen=final_gen)
            with open(final_agent,'wb') as w:
                pickle.dump(dual_MCTS_Agent,w)

    if not args.single_process or args.ligpre:
        # 生成分子的ligpre
        ligpre(os.path.abspath(iter_csv),os.path.abspath(iter_mae),args.nchir,ligpre_path=args.ligpre_path,device=device)
        for i in trange(100000):
            time.sleep(10)
            if os.path.exists(iter_mae):
                break
    
    if not args.single_process or args.docking:
        # 生成分子的docking
        docking(os.path.abspath(iter_mae),os.path.abspath(grid_5ntp),glide_path=args.glide_path,prec=prec,path=os.path.abspath(docking_dir),device=device)
        time.sleep(10)
        docking(os.path.abspath(iter_mae),os.path.abspath(grid_6qu7),glide_path=args.glide_path,prec=prec,path=os.path.abspath(docking_dir),device=device)
        for i in trange(100000):
            time.sleep(10)
            if os.path.exists(mae_5ntp) and os.path.exists(mae_6qu7):
                break
        schrodinger_run=os.path.join(args.schrodinger_dir,'run')
        os.system(f'{schrodinger_run} python3 utils/mae_to_csv.py '
                f'--task get_scores --info_csv {iter_csv} '
                f'--input_maes {mae_5ntp},{mae_6qu7} '
                f'--output {docking_csv}')
        df_next=pd.read_csv(docking_csv)
        if iter==0:
            df_next.to_csv(next_csv,index=False)
        elif iter>=1:
            df=pd.DataFrame({'SMILES':np.concatenate([df_last['SMILES'],df_next['SMILES']]),'Title':np.concatenate([df_last['Title'],df_next['Title']]),
                            f'{pdb_id1}_{prec}':np.concatenate([df_last[f'{pdb_id1}_{prec}'],df_next[f'{pdb_id1}_{prec}']]),
                            f'{pdb_id2}_{prec}':np.concatenate([df_last[f'{pdb_id2}_{prec}'],df_next[f'{pdb_id2}_{prec}']])})
            df.to_csv(next_csv,index=False)
        os.system(f'mv {mae_5ntp} {new_5ntp}')
        os.system(f'mv {mae_6qu7} {new_6qu7}')
    
    if not args.single_process or args.training:
        # 生成分子的docking结果的训练
        train(next_csv,new_model,pdb_id1,pdb_id2,prec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='rorgt_dhodh')
    parser.add_argument('--cache_dir',default='data/temp_data/')
    parser.add_argument('--generated_dir',default='data/outputs/generated/')
    parser.add_argument('--model_dir',default='data/models/dgl')
    parser.add_argument('--gen_num',type=int,default=10000)
    parser.add_argument('--iter',default='0,1,2,3,4')
    parser.add_argument('--ligpre_dir',default='/public/home/chensheng/project/aixfuse2/data/outputs/ligpre/')
    parser.add_argument('--nchir',default=32)
    parser.add_argument('--grid_dir',default='data/inputs/target_structures/grids/')
    parser.add_argument("--single_process", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--ligpre", action="store_true")
    parser.add_argument("--docking", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument('--schrodinger_dir', required=True)
    parser.add_argument('--ligpre_path', required=True)
    parser.add_argument('--glide_path', required=True)


    args = parser.parse_args()
    for iter in args.iter.split(','):
        iter=int(iter)
        iter_gen(args,iter)