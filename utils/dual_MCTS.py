from rdkit import Chem
from rdkit.Chem import AllChem
import random
import numpy as np
from utils_chem import fuse,extract_core_func,scoring_func
from utils_common import get_properties
import sascorer
import math
from copy import deepcopy
from multiprocessing import Pool
# from docking_NN_pred import get_predictor,predict
# from moses.metrics import weight, SA, QED,mol_passes_filters
from tqdm import trange
import joblib
from dgllife.model import model_zoo
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from torch.utils.data import DataLoader
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import mol_to_bigraph
import dgl
import torch
import pandas as pd
from functools import partial
import pickle

def norm_mol(mol):
    try:
        smiles = Chem.MolToSmiles(mol,isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None
    

random.seed(2023)
NCPU=64 # 并行核数
pool=Pool(NCPU) # 并行进程池
W1=1 # docking score1的分数权重
W2=1 # docking score2的分数权重
INIT_Q=0

class rf_model:
    def __init__(self,clf_path):
        self.clf = joblib.load(clf_path)

    def __call__(self, mol_list):
        fps = []
        ind_list=[]
        for i,mol in enumerate(mol_list):
            try:
                smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
                norm_mol=Chem.MolFromSmiles(smiles)
                fp = AllChem.GetMorganFingerprintAsBitVect(norm_mol,2, nBits=2048)
                fps.append(fp)
                ind_list.append(i)
            except:
                pass
        scores=[0 for _ in mol_list]
        # fps = np.concatenate(fps, axis=0)
        if fps:
            s = self.clf.predict_proba(fps)[:, 1]
        for i,ind in enumerate(ind_list):
            scores[ind]=s[i]
        return np.float32(scores)

# model1=joblib.load('models/rf_models/svm_rorgt_score.pkl')
# model2=joblib.load('models/rf_models/svm_dhodh_score.pkl')


def collate_molgraphs(data):
    smiles_list, graph_list = map(list, zip(*data))
    
    bg = dgl.batch(graph_list)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return smiles_list, bg

class GraphDataset(object):
    def __init__(self,smiles_list,smiles_to_graph):
        self.smiles=smiles_list
        if len(smiles_list) > 100:
            self.graphs = pool.map(smiles_to_graph,self.smiles)
        else:
            self.graphs = []
            for s in self.smiles:
                self.graphs.append(smiles_to_graph(s))
        

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)

class MTATFP_model:
    def __init__(self,clf_path,device='cuda',n_tasks=2):
        self.atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
        self.bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
        n_feats = self.atom_featurizer.feat_size('hv')
        e_feats = self.bond_featurizer.feat_size('he')
        self.n_tasks=n_tasks
        model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                        edge_feat_size=e_feats,
                                        num_layers=2,
                                        num_timesteps=1,
                                        graph_feat_size=300,
                                        n_tasks=n_tasks
                                            )
        model.load_state_dict(torch.load(clf_path,map_location=torch.device(device)))
        self.device=device
        self.gcn_net = model.to(device)
    
    def __call__(self, mol_list,get_node_weight=False):
        new_mols=pool.map(norm_mol, mol_list)
        i=0
        new_mol_list=[]
        ind_list=[]
        for mol in new_mols:
            if mol is not None:
                new_mol_list.append(mol)
                ind_list.append(i)
            i+=1
        final_scores=np.ones([len(mol_list),self.n_tasks])
        if ind_list:
            mol_to_graph = partial(mol_to_bigraph, node_featurizer=self.atom_featurizer,edge_featurizer=self.bond_featurizer)
            test_datasets=GraphDataset(new_mol_list,mol_to_graph)
            test_loader = DataLoader(test_datasets, batch_size=256,shuffle=False,collate_fn=collate_molgraphs)
            results=[]
            for batch_data in test_loader:
                test_smiles, test_bg = batch_data
                test_bg = test_bg.to(self.device)
                test_n_feats = test_bg.ndata.pop('hv').to(self.device)
                test_e_feats = test_bg.edata.pop('he').to(self.device)
                if get_node_weight:
                    test_prediction,node_weight = self.gcn_net(test_bg, test_n_feats, test_e_feats,get_node_weight=get_node_weight)
                else:
                    test_prediction = self.gcn_net(test_bg, test_n_feats, test_e_feats)
                result=test_prediction.detach().cpu().numpy()
                result=-result
                result[result<1]=1
                result[result>20]=20 # 12?
                results.append(result)
            scores=np.concatenate(results,axis=0)
            final_scores[ind_list]=scores
        if get_node_weight:
            return (final_scores[:,0],final_scores[:,1]),node_weight
        else:
            return final_scores[:,0],final_scores[:,1]


# 化学空间探索记录类，其对象在DualMCTS类中通过explore方法产生，储存为列表形式
class ExploredRecord:
    def __init__(self,node1,node2):
        self.node1=node1 # 蒙特卡洛树节点1，为Rationale类
        self.node2=node2 # 蒙特卡洛树节点2，为Rationale类
        self.mol=fuse(node1.mol,node2.mol,999,True) # 融合节点1与节点2的分子
        self.mol.SetProp('SMILES1',node1.smiles)
        self.mol.SetProp('SMILES2',node2.smiles)
        title='|'.join([self.node1.name,self.node2.name])
        self.mol.SetProp('_Name',f'AIxFuse|{title}')
        
    
    # 记录融合后分子性质，score_items参数通过eval mol得到
    def get_score_items(self,score_items): 
        self.score_items=score_items
        self.base_score_items=score_items

    # 更新融合后分子docking score，未来可以用于其他性质更新
    def update_score_items(self,item_dict,model_type='NN'):
        score=1
        harmonic_mean=0
        for key in item_dict:
            value = item_dict[key]
            self.score_items[key]=value
            if model_type=='NN':
                score*=max(0.1,value-6)
            else:
                score*=max(0.01,min(value,0.7)*10-5)

            # score+=max(1,value-4)
            # harmonic_mean+=1/max(1,value-4)

        self.score_items['total_score']=self.base_score_items['total_score']*score
    
# 蒙特卡洛树节点类，作为MCTS_Agent，Core，CoreAnchor，SideAnchor,Rationale的父类
class MCT_Node:
    def __init__(self,parent):
        self.parent=parent # 父节点
        self.name=''
        self.children=[] # 子节点
        self.weights=[] # 子节点随机采样权重
        self.W=0 # 总探索得分
        self.N=0 # 总探索次数
        self.P=INIT_Q # 本次模拟得分
        self.num_rationales=0
        self.records=[]
    
    # 随机采样
    def sample_random(self):
        return random.choices(self.children, weights=self.weights, k=1)[0]

    # 计算该节点平均探索得分
    def Q(self):
        return np.percentile(self.records,90) if len(self.records) > 0 else self.P

    # 计算该节点模拟得分，以已探索次数的相反数加权
    def U(self, n):
        return self.parent.C_PUCT / self.C_PUCT  * self.P * math.sqrt(n) / math.sqrt(1 + self.N)
    
    # 采样最优子节点
    def sample(self):
        if random.random() >= (self.N)/len(self.children):
            return self.sample_random()
        else:
            return self.sample_best()
    
    def sample_best(self):
        return max(self.children, key=lambda x : x.Q() + x.U(self.N+1))

    # 自下而上更新W和N
    def update(self,score):
        self.records.append(score)
        # self.W+=score
        # self.N+=1
        self.P=score
        if self.parent:
            self.parent.update(score)

    # 自上而下清除W和N
    def clear(self):
        self.records=[]
        # self.W=0
        # self.N=0
        self.P=INIT_Q
        for child in self.children:
            child.clear()

    def get_C_PUCT(self):
        self.C_PUCT=1/np.sqrt(self.num_rationales) # 可以调控蒙特卡洛树的探索倾向
        for child in self.children:
            child.get_C_PUCT()

    def update_num(self):
        self.num_rationales+=1
        if self.parent:
            self.parent.update_num()

# 双蒙特卡洛树搜索代理类
class DualMCTS():
    def __init__(self,mol_dict1,mol_dict2):
        self.sampler1=MCTS_Agent(mol_dict1) # 蒙特卡洛树搜索代理1，用于采样Target1的Core
        self.sampler2=MCTS_Agent(mol_dict2) # 蒙特卡洛树搜索代理2，用于采样Target2的Core
        self.gen_res={} # 当前已探索记录的列表
        self.sim_res={} # 当前已模拟记录的列表
        
    
    def init_explore(self,n_sample=300000,max_atoms=50):
        mol_list=[]
        rationale_pairs=set()
        smiles_set=set()
        smiles_list=[]
        title_list=[]
        SA_list=[]
        pbar=trange(n_sample*10)
        r1_list=[]
        r2_list=[]
        r1_smiles_list=[]
        r2_smiles_list=[]
        core1_smiles_list=[]
        core2_smiles_list=[]
        for i in pbar:
            pbar.set_postfix({'generated:':len(mol_list)})
            if len(mol_list)>=n_sample:
                break
            rationale1=self.sampler1.sample_random().sample_random().sample_random().sample_random()
            rationale2=self.sampler2.sample_random().sample_random().sample_random().sample_random()
            if (rationale1,rationale2) in rationale_pairs:
                continue
            rationale_pairs.add((rationale1,rationale2))
            if len(rationale1.mol.GetAtoms())+ len(rationale2.mol.GetAtoms())>max_atoms:
                continue
            mol1=deepcopy(rationale1.mol)
            mol2=deepcopy(rationale2.mol)
            mol=fuse(mol1,mol2,999,True)
            
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
            if smiles not in smiles_set:
                smiles_set.add(smiles)
                new_mol=Chem.MolFromSmiles(smiles)
                # try:
                #     qed = QED(new_mol)
                #     if qed <= 0.1:
                #         continue
                #     sa = SA(new_mol)
                #     if  sa >= 4:
                #         continue
                # except:
                #     continue
                mol_list.append(new_mol)
                smiles_list.append(smiles)
                title='|'.join([rationale1.name,rationale2.name])
                title_list.append(f'AIxFuse|{title}')
                # SA_list.append(sa)
                r1_list.append(rationale1.mol)
                r2_list.append(rationale2.mol)
                r1_smiles_list.append(rationale1.smiles)
                r2_smiles_list.append(rationale2.smiles)
                core1_smiles_list.append(rationale1.parent.parent.parent.smiles)
                core2_smiles_list.append(rationale2.parent.parent.parent.smiles)
                i+=1
        df=pd.DataFrame({'SMILES':smiles_list, 'Title':title_list, 
                         'Rationale1':r1_smiles_list,'Rationale2':r2_smiles_list,
                         'Core1':core1_smiles_list,'Core2':core2_smiles_list})
        df=get_properties(df)
        return df

    # TODO 根据docking结果更新记录，包括W,N,simulated和gen_res等
    def update_records(self,docking_records):
        self.sampler1.clear()
        self.sampler2.clear()
        for title in docking_records:
            if title in self.gen_res:
                record = docking_records[title]
                gen_record=self.gen_res[title]
                gen_record.update_score_items(record,self.model_type)
                gen_record.node1.update(gen_record.score_items['total_score']) # 更新W和N
                gen_record.node2.update(gen_record.score_items['total_score']) # 更新W和N
        if self.final_gen:
            self.gen_res={}
    
    def get_p(self,ref_data1,ref_data2):
        self.eval_stats={}
        for item in ['Weight','LogP','RotBond']:
            self.eval_stats[item]={}
            self.eval_stats[item]['min']=min(np.percentile(ref_data1[item],10),np.percentile(ref_data2[item],10))
            self.eval_stats[item]['max']=max(np.percentile(ref_data1[item],90),np.percentile(ref_data2[item],90))
            self.eval_stats[item]['std']=max(np.std(ref_data1[item]),np.std(ref_data2[item]))

    # TODO 迭代探索，每模拟n个self.sim_res，docking并进行NN训练，再根据docking结果更新记录
    def iter_explore(self,n_sample,model_path,docking_result,gen_csv,final_gen=False,model_type='NN'):
        self.gen_smis=set()
        df1=pd.read_csv('data/compare/rorgt_dhodh/rorgt.csv')
        df2=pd.read_csv('data/compare/rorgt_dhodh/dhodh.csv')
        self.get_p(df1,df2)
        
        if model_type=='NN':
            self.mtatfp_model=MTATFP_model(model_path)
        else:
            model1_path,model2_path=model_path.split(',')
            self.model1=rf_model(model1_path)
            self.model2=rf_model(model2_path)
        self.eval=partial(scoring_func,stats=self.eval_stats,model_type=model_type)
        self.model_type=model_type
        tbar=trange(n_sample)
        self.final_gen=final_gen
        if docking_result:
            self.update_records(docking_result)
            self.title_set=set(docking_result.keys())
        else:
            self.title_set=set()
        self.last_num=len(self.title_set)
        self.n_sample=n_sample

        self.gen_csv=gen_csv
        with open(self.gen_csv,'w') as w:
            w.write('SMILES,Title,Rationale1,Rationale2,Core1,Core2,SA,QED,Weight,LogP,RotBond,Filtered,docking_pred1,docking_pred2,total_score\n')
            # w.write('SMILES,Title,docking1,docking2,SA,QED,Weight,LogP,RotBond,Filtered,SMILES1,SMILES2,total_score\n')
        
        for i in tbar:
            if len(self.title_set)>=self.last_num+self.n_sample:
                break
            # if i % 5000 ==0:
            #     with open(f'data/temp_data/dual_MCTS_Agent_5NTP_6QU7_{i}.pkl','wb') as p:
            #         pickle.dump(self,p)
            
            res=self.explore()
            if final_gen:
                with open(self.gen_csv,'a') as w:
                    w.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(res.score_items['SMILES'],
                                                                res.mol.GetProp('_Name'),
                                                                res.mol.GetProp('SMILES1'),
                                                                res.mol.GetProp('SMILES2'),
                                                                res.node1.parent.parent.parent.smiles,
                                                                res.node2.parent.parent.parent.smiles,
                                                                res.score_items['SA'],
                                                                res.score_items['QED'],
                                                                res.score_items['Weight'],
                                                                res.score_items['LogP'],
                                                                res.score_items['RotBond'],
                                                                res.score_items['pass_filter'],
                                                                res.score_items['docking1'],
                                                                res.score_items['docking2'],
                                                                res.score_items['total_score']))
            self.gen_smis.add(res.score_items['SMILES'])
            try:
                tbar.set_postfix({'score':'%.2f'%(res.score_items['total_score']),
                                'docking1':'%.2f'%(res.score_items['docking1']),
                                'docking2':'%.2f'%(res.score_items['docking2']),
                                'QED':'%.2f'%(res.score_items['QED']),
                                'SA':'%.2f'%(res.score_items['SA']),
                                'unique':'%.2f'%(len(self.gen_smis)/(i+1)),
                                'num_res':'%d'%(len(self.title_set)-self.last_num)})
            except:
                pass

    # 探索 TODO 进行提速，比如并行模拟
    def explore(self):
        record,sim_res,scores=self.simulate(self.sampler1,self.sampler2,[],[]) # 模拟
        arg_sort=np.argsort(scores)[::-1]
        if not self.final_gen:
            to_write=[]
            for ind in arg_sort:
                res=sim_res[ind]
                self.sim_res[res.mol.GetProp('_Name')]=res
                if len(self.title_set)>=self.last_num+self.n_sample:
                    break
                
                if res.mol.GetProp('_Name') not in self.title_set:
                    if not to_write:
                        to_write.append(res)
                        self.title_set.add(res.mol.GetProp('_Name'))
                    elif  res.score_items['QED'] is not None and res.score_items['QED'] >= 0.2 \
                            and res.score_items['SA'] is not None and res.score_items['SA'] < 4 :
                        if random.random()<0.1:
                            to_write.append(res)
                            self.title_set.add(res.mol.GetProp('_Name'))
            for res in to_write:
                with open(self.gen_csv,'a') as w:
                    w.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(res.score_items['SMILES'],
                                                                    res.mol.GetProp('_Name'),
                                                                    res.mol.GetProp('SMILES1'),
                                                                    res.mol.GetProp('SMILES2'),
                                                                    res.node1.parent.parent.parent.smiles,
                                                                    res.node2.parent.parent.parent.smiles,
                                                                    res.score_items['SA'],
                                                                    res.score_items['QED'],
                                                                    res.score_items['Weight'],
                                                                    res.score_items['LogP'],
                                                                    res.score_items['RotBond'],
                                                                    res.score_items['pass_filter'],
                                                                    res.score_items['docking1'],
                                                                    res.score_items['docking2'],
                                                                    res.score_items['total_score']))
        for ind in arg_sort:
            record=sim_res[ind]
            if record.mol.GetProp('_Name') not in self.gen_res:
                break
        record.node1.update(record.score_items['total_score']) # 更新W和N
        record.node2.update(record.score_items['total_score']) # 更新W和N
        self.gen_res[record.mol.GetProp('_Name')]=record
        
        return record

    # 模拟函数，递归模拟，每层递归取模拟最优子节点
    def simulate(self,node1,node2,sim_res,scores):
        if isinstance(node1,Rationale): # 如果递归到Rationale层，则结束递归，并记录模拟结果
            record=ExploredRecord(node1,node2)
            # scores1=self.model1([record.mol])
            # scores2=self.model2([record.mol])
            if self.model_type=='NN':
                scores1,scores2=self.mtatfp_model([record.mol])
            else:
                scores1=self.model1([record.mol])
                scores2=self.model2([record.mol])
            # score1=predict([record.mol], 'cuda:3',self.model1)[0]
            # score2=predict([record.mol], 'cuda:3',self.model2)[0]
            record.get_score_items(self.eval(record.mol))
            record.update_score_items({'docking1':scores1[0],'docking2':scores2[0]},self.model_type)
            sim_res.append(record)
            scores.append(record.score_items['total_score'])
            return record,sim_res,scores
        mol_list=[] # 融合分子列表
        candidates=[] # 候选节点列表

        # 靶点1，随机采样靶点2的rationale，然后融合两个分子，放入融合分子列表和候选节点列表
        ref_node2=node2
        while not isinstance(ref_node2,Rationale):
            ref_node2=ref_node2.sample()
        # self.sampler2.sample_random().sample_random().sample_random().sample_random()
        for child1 in node1.children:
            this_child=child1
            while not isinstance(this_child,Rationale):
                this_child=this_child.sample()
            record = ExploredRecord(this_child,ref_node2)
            mol_list.append(record.mol)
            candidates.append((child1,record))

        # 靶点2
        ref_node1=node1
        while not isinstance(ref_node1,Rationale):
            ref_node1=ref_node1.sample()
        # ref_node1=self.sampler1.sample_random().sample_random().sample_random().sample_random()
        for child2 in node2.children:
            this_child=child2
            while not isinstance(this_child,Rationale):
                this_child=this_child.sample()
            record = ExploredRecord(ref_node1,this_child)
            mol_list.append(record.mol)
            candidates.append((child2,record))
        # 分子评估
        # scores1=predict(mol_list, 'cuda:3',self.model1) # docking score NN预测
        # scores2=predict(mol_list, 'cuda:3',self.model2)

        if len(mol_list)>0:
            if self.model_type=='NN':
                scores1,scores2=self.mtatfp_model(mol_list)
            else:
                scores1=self.model1(mol_list)
                scores2=self.model2(mol_list)
            if len(mol_list)>100: # 若数量大，多进程评估分子性质
                results = pool.map(self.eval, mol_list)
            else:
                results=[]
                for mol in mol_list:
                    results.append(self.eval(mol))
            for score_items,out,score1,score2 in zip(results,candidates,scores1,scores2):
            # for score_items,out in zip(results,candidates):
                node,record=out
                record.get_score_items(score_items)
                record.update_score_items({'docking1':score1,'docking2':score2},self.model_type)
                sim_res.append(record)
                scores.append(record.score_items['total_score'])
                node.P = record.score_items['total_score']
                # _child1.update(record.score_items['total_score']) # 更新W和N
                # _child2.update(record.score_items['total_score']) # 更新W和N

        # 采样最优分子
        best_child1=node1.sample_best()
        best_child2=node2.sample_best()
        return self.simulate(best_child1,best_child2,sim_res,scores)
    
# 蒙特卡洛树搜索代理类
class MCTS_Agent(MCT_Node):
    def __init__(self,mol_dict):
        super().__init__(None)
        self.mol_dict=mol_dict # 从struct_ana中获取的活性分子core，frag，ic50等信息
        self.rationale_count=0 # 记录该靶点rationale数量
        self.core_anchor_count=0 # 记录该靶点core_anchor数量
        self.side_anchor_count=0 # 记录该靶点side_anchor数量
        self.get_children()
        self.get_C_PUCT()
        
        
    # 获取子节点Core对象列表
    def get_children(self):
        for smiles in self.mol_dict:
            core = Core(self,smiles,self.mol_dict[smiles],f'C{len(self.children)}')
            if len(core.children) >0:
                self.children.append(core)
                self.weights.append(core.total_pic50)
            else:
                print('empty core children!',core.smiles)

# Core类
class Core(MCT_Node):
    def __init__(self,parent,smiles,mol_infos,name):
        super().__init__(parent)
        self.name='_'.join([self.parent.name,name])
        self.smiles=smiles
        self.mol_infos=mol_infos
        self.mol=mol_infos['core']
        self.atom_num=len(self.mol.GetAtoms())
        self.act_num=len(self.mol_infos['act_mols'])
        self.total_pic50=-sum(self.mol_infos['iec50'])
        self.core_anchors={}
        self.get_children()
        self.get_r_group_combs()
        self.get_side_anchor_children()

    # 获取子节点CoreAnchor对象列表
    def get_children(self):
        for site in self.mol_infos:
            if isinstance(site,int):
                self.children.append(CoreAnchor(self,site))
                self.weights.append(sum([min(len(mol.GetAtoms()),8) for mol in self.mol_infos[site]]))
    
    # 根据子节点CoreAnchor的R_Groups（SideAnchor和SideChain）获取R_Group组合列表
    def get_r_group_combs(self):
        for i,child1 in enumerate(self.children):
            r_group_combs=[[]]
            r_group_comb_weights=[1]
            for j,child2 in enumerate(self.children):
                if i!=j:
                    temp_combs=[]
                    temp_weights=[]
                    for r_group_comb, comb_weight in zip(r_group_combs,r_group_comb_weights):
                        for r_group, group_weight in zip(child2.r_groups,child2.r_group_weights):
                            new_comb=r_group_comb+[r_group]
                            temp_combs.append(new_comb)
                            temp_weights.append(comb_weight*group_weight)
                    r_group_combs=temp_combs
                    r_group_comb_weights=temp_weights
            sort_idx=np.argsort(-np.array(r_group_comb_weights))[:len(child1.side_infos)+1] # R_Group的数量不超过该CoreAnchor的侧链数量
            child1.r_group_combs=[r_group_combs[i] for i in sort_idx]
            child1.r_group_comb_weights=[r_group_comb_weights[i] for i in sort_idx]
    
    # 根据R_Group组合列表获取SideAnchor的子节点
    def get_side_anchor_children(self):
        for core_anchor in self.children:
            for side_anchor in core_anchor.children:
                for i,(r_group_comb,r_group_comb_weight) in enumerate(zip(core_anchor.r_group_combs,core_anchor.r_group_comb_weights)):
                    side_anchor.children.append(Rationale(side_anchor,r_group_comb,f'R{i}'))
                    side_anchor.weights.append(r_group_comb_weight)

# CoreAnchor类
class CoreAnchor(MCT_Node):
    def __init__(self,parent,site):
        super().__init__(parent)
        self.name='_'.join([self.parent.name,f'A{site}'])
        self.site=site
        self.side_infos=self.parent.mol_infos[site]
        self.r_groups=[]
        self.r_group_weights=[]
        self.get_children()
        self.parent.parent.core_anchor_count+=1
        
    # 获取子节点SdieAnchor对象列表
    def get_children(self):
        # 先获取该Anchor连接的SideChain信息
        side_chains={}
        for frag in self.side_infos:
            frag_smiles=Chem.MolToSmiles(frag,isomericSmiles=False)
            if frag_smiles not in side_chains:
                side_chains[frag_smiles]=SideChain(self,frag,frag_smiles)
            side_chains[frag_smiles].count+=1
        
        # 根据SideChain类的方法获取所有可merge的SideAnchor及其权重，同时获取可以作为R_Group的SideChain/SideAnchor作为R_Group的权重
        side_chains['side_anchors']={}
        for frag_smiles in side_chains:
            if frag_smiles=='side_anchors':
                continue
            side_chain=side_chains[frag_smiles]
            # if side_chain.atom_num<=5:# 如果SideChain原子数小于等于5，可作为R_Group
            self.r_groups.append(side_chain)
            self.r_group_weights.append(side_chain.count)
            for side_anchor in side_chain.anchors:
                if side_anchor.anchor_smiles not in side_chains['side_anchors']:
                    side_chains['side_anchors'][side_anchor.anchor_smiles]=side_anchor
                    self.parent.parent.side_anchor_count+=1
                if side_anchor.anchor_smiles=='':
                    # side_chains['side_anchors'][side_anchor.anchor_smiles]=self.parent.act_num
                    side_anchor.anchor_wegiht=self.parent.act_num
                    side_anchor.r_group_weight=self.parent.act_num-len(self.side_infos)+1
                else:
                    side_anchor.anchor_wegiht=max(side_chain.count,side_anchor.anchor_wegiht)
                    side_anchor.r_group_weight+=side_chain.count#max(side_chain.count*side_anchor.atom_num/side_chain.atom_num,side_anchor.r_group_weight)
        
        # 综合信息获取子节点和R_Groups
        for anchor_smiles in side_chains['side_anchors']:
            side_anchor=side_chains['side_anchors'][anchor_smiles]
            side_anchor.name=self.name+f'S{len(self.children)}'
            self.children.append(side_anchor)
            self.weights.append(side_anchor.anchor_wegiht)#1)
            if side_anchor.atom_num<=5 : # 如果SideAnchor原子数小于等于5，可作为R_Group
                self.r_groups.append(side_anchor)
                self.r_group_weights.append(side_anchor.r_group_weight)

# SideChain类，作为中间类，用于产生SideAnchor
class SideChain():
    def __init__(self,core_anchor,mol,smiles):
        self.parent=core_anchor
        self.smiles=smiles
        self.mol=mol
        self.r_group_mol=self.mol
        self.atom_num=len(self.mol.GetAtoms())
        self.count=0
        anchors=self.get_anchors()
        self.anchors=anchors# if len(anchors)<=20 else [anchors[0]]+random.sample(anchors,20) #如果锚点太多，随机选十个
        
    # 产生可用SideAnchor
    def get_anchors(self):
        start_atom=-1
        for atom in self.mol.GetAtoms():
            if atom.GetAtomMapNum()!=0:
                start_atom=atom.GetIdx()
                break
        assert start_atom != -1

        anchors=[SideAnchor(self.parent,-1,None,None)]
        for atom in self.mol.GetAtoms():
            idx=atom.GetIdx()
            tmp_mol=Chem.RWMol(self.mol)
            for tmp_atom in tmp_mol.GetAtoms():
                tmp_atom.SetAtomMapNum(tmp_atom.GetIdx())
            tmp_mol.RemoveAtom(idx)
            frags=Chem.GetMolFrags(tmp_mol.GetMol(), asMols=True, sanitizeFrags=False)
            for frag in frags:
                flag=True
                to_del=[]
                for f in frag.GetAtoms():
                    idx2=f.GetAtomMapNum()
                    to_del.append(idx2)
                    if start_atom == idx2 or self.mol.GetRingInfo().AreAtomsInSameRing(idx,idx2):
                        flag=False
                if flag:
                    side_chain_mol=deepcopy(self.mol)
                    side_chain_mol.GetAtomWithIdx(idx).SetAtomMapNum(999)
                    anchor_mol=extract_core_func(side_chain_mol, set(range(self.atom_num))-set(to_del))
                    if anchor_mol:
                        Chem.SanitizeMol(anchor_mol)
                        anchors.append(SideAnchor(self.parent,idx,anchor_mol,self.mol))
                    else:
                        print('anchor mol error!')
        return anchors
    
# SideAnchor类，
class SideAnchor(MCT_Node):
    def __init__(self,parent,site,anchor_mol,side_chain_mol):
        super().__init__(parent)
        self.name=None
        self.side_chain_mol=side_chain_mol
        self.anchor_wegiht=0
        self.r_group_weight=0
        self.site=site
        self.anchor_mol=anchor_mol
        self.r_group_mol=self.get_r_group_mol()
        if anchor_mol:
            self.anchor_smiles=Chem.MolToSmiles(anchor_mol,isomericSmiles=False)
            self.atom_num=len(anchor_mol.GetAtoms())
        else:
            self.anchor_smiles=''
            self.atom_num=0
        self.mol=self.get_mol()

    # 融合Core和SideAnchor，得到Rationale骨架
    def get_mol(self):
        core_mol = deepcopy(self.parent.parent.mol)
        if self.site==-1:
            core_mol.GetAtomWithIdx(self.parent.site).SetAtomMapNum(999)
            return core_mol
        else:
            core_side=fuse(core_mol,self.anchor_mol,self.parent.site+1)
            if core_side:
                return core_side
            else:
                print('not fuse')
        return None
    
    # 得到用于R_Group的分子
    def get_r_group_mol(self):
        if self.anchor_mol:
            mol=deepcopy(self.anchor_mol)
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum()==self.parent.site+1:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomMapNum()==999:
                            atom.SetAtomMapNum(0)
                    break
            return mol
        return None

# Rationale类
class Rationale(MCT_Node):
    def __init__(self,parent,r_group_comb,name):
        super().__init__(parent)
        self.name='_'.join([self.parent.name,name])
        self.r_group_comb=r_group_comb
        self.parent.parent.parent.parent.rationale_count+=1
        self.mol=self.get_mol()
        self.act_atom=len(self.mol.GetAtoms())
        mol=deepcopy(self.mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        self.smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
        self.update_num()
    
    # 迭代融合Rationale骨架和R_Group
    def get_mol(self):
        mol = deepcopy(self.parent.mol)
        for r_group in self.r_group_comb:
            if r_group.r_group_mol:
                r_group_mol = deepcopy(r_group.r_group_mol)
                tmp_mol=fuse(mol,r_group_mol,r_group.parent.site+1,clear=True)
                if tmp_mol:
                    mol=tmp_mol
                else:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomMapNum()==r_group.parent.site+1:
                            atom.SetAtomMapNum(0)
                    print('not fuse')
            else:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum()==r_group.parent.site+1:
                        atom.SetAtomMapNum(0)
        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
        return mol
    