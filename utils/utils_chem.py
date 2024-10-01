from rdkit import Chem
import rdkit
import sascorer

from rdkit.Chem import Lipinski,AllChem,DataStructs
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
from functools import partial
import pandas as pd


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def calc_sim_datas2refs(s,ref_fps):
    result=[]
    fp =AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,1024)
    return DataStructs.BulkTanimotoSimilarity(fp,ref_fps)

def hit_design(ref,input,):
    df_ref=pd.read_csv(ref)
    df_input=pd.read_csv(input)
    ref_fps=[AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,1024) for s in df_ref['SMILES']]
    
    data=[]
    for s in df_input['SMILES']:
        data.append(s)
    
    work_func = partial(calc_sim_datas2refs,ref_fps=ref_fps)
    pool = Pool(64)
    results = pool.map(work_func, data)

    score_list=[]
    title_list=[]
    for result in results:
        score_list.append(np.max(result))
        title_list.append(df_ref['Title'][np.argmax(result)])
    name=ref.split('/')[-1].split('.')[0]
    df_input[f'sim2d_{name}']=score_list
    df_input[f'sim2d_{name}_match']=title_list
    df_input.to_csv(input,index=False)


def calc_dis(coordList1,coordList2):
    y = [coordList2 for _ in coordList1]
    y = np.array(y)
    x = [coordList1 for _ in coordList2]
    x = np.array(x)
    x = x.transpose(1, 0, 2)
    a = np.linalg.norm(np.array(x) - np.array(y), axis=2)
    return a

def fuse(cur_mol,side_chain_mol,atom_map_num,clear=False):
    fused_mol = Chem.CombineMols(cur_mol,side_chain_mol)
    frags = Chem.GetMolFrags(fused_mol, asMols=False)
    assert len(frags)==2

    fused_mol=Chem.RWMol(fused_mol)
    for i in frags[0]:
        atom1=fused_mol.GetAtomWithIdx(i)
        if atom1.GetAtomMapNum() ==atom_map_num:
            break
    if atom1.GetAtomMapNum() !=atom_map_num:
        for i in frags[0]:
            atom1=fused_mol.GetAtomWithIdx(i)
            if atom1.GetAtomMapNum() ==999:
                break
        if atom1.GetAtomMapNum() !=999:
            return None

    for j in frags[1]:
        atom2=fused_mol.GetAtomWithIdx(j)
        if atom2.GetAtomMapNum() == atom_map_num:
            break
    if atom2.GetAtomMapNum() !=atom_map_num:
        for j in frags[1]:
            atom2=fused_mol.GetAtomWithIdx(j)
            if atom2.GetAtomMapNum() == 999:
                break
        if atom2.GetAtomMapNum() !=999:
            return None
    
    try:
        fused_mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
    except:
        print('error in fuse!:',atom_map_num,i,j,Chem.MolToSmiles(fused_mol))
        return None
    atom1.SetNumExplicitHs(max(atom1.GetNumExplicitHs()-1,0))
    atom2.SetNumExplicitHs(max(atom2.GetNumExplicitHs()-1,0))
    if clear or atom1.GetAtomMapNum() !=999:
        atom1.SetAtomMapNum(0)
    if clear or atom2.GetAtomMapNum() !=999:
        atom2.SetAtomMapNum(0)
    return fused_mol.GetMol()

# def expand_atoms(reserved_atoms,mol):
#     new_atom=reserved_atoms.copy()
#     for i in range(len(mol.GetAtoms())):
#         for a in reserved_atoms:
#             if mol.GetRingInfo().AreAtomsInSameRing(i,a):
#                 new_atom.add(i)
#     if new_atom>reserved_atoms:
#         reserved_atoms=new_atom
#         return expand_atoms(reserved_atoms,mol)
#     return reserved_atoms

def dfs_get_ring(path_list,mol):
    atom_set=set()
    got_ring=False
    for path in path_list:
        for atom_idx in path:
            atom_set.add(atom_idx)
    new_atom_set=atom_set.copy()
    for path in path_list:
        neigh_set=set()
        for atom_idx in path:
            for nei in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nei.GetIdx() not in new_atom_set:
                    neigh_set.add(nei.GetIdx())
        for nei_idx in neigh_set:
            if mol.GetAtomWithIdx(nei_idx).IsInRing():
                got_ring=True
            path.append(nei_idx)
            new_atom_set.add(nei_idx)
    if len(atom_set)==len(new_atom_set) or got_ring:
        return path_list
    return dfs_get_ring(path_list,mol)

def expand_atoms(reserved_atoms,mol):
    new_atom=reserved_atoms.copy()
    for atom_idx in reserved_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for nei in atom.GetNeighbors():
            n=nei.GetIdx()
            if n not in new_atom:
                bond=mol.GetBondBetweenAtoms(n,atom_idx)
                if bond.GetBondType()!=Chem.rdchem.BondType.SINGLE or bond.IsInRing():
                    new_atom.add(n)
    if new_atom>reserved_atoms:
        return expand_atoms(new_atom,mol)
    ring_flag=False
    for atom_idx in reserved_atoms:
        if mol.GetAtomWithIdx(atom_idx).IsInRing():
            ring_flag=True
            break
    path_list=[[atom_idx] for atom_idx in reserved_atoms]
    if not ring_flag:
        path_list= dfs_get_ring(path_list,mol)
        new_atom=set()
        for path in path_list:
            ring_flag=False
            for atom_idx in path:
                if mol.GetAtomWithIdx(atom_idx).IsInRing():
                    ring_flag=True
            if ring_flag:
                for atom_idx in path:
                    new_atom.add(atom_idx)
            else:
                new_atom.add(path[0])
        return expand_atoms(new_atom,mol)
    
    return reserved_atoms

def is_flex(mol):
    flag=False
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if len(a1.GetNeighbors())==2 and len(a2.GetNeighbors())==2:
                if not a1.GetBonds()[0].IsInRing() and not a1.GetBonds()[1].IsInRing()\
                    and not a2.GetBonds()[0].IsInRing() and not a2.GetBonds()[1].IsInRing():
                    flag=True
                    break
    return flag

# 评估分子性质（除docking score外）
def eval(mol):
    from moses.metrics import weight, logP, SA, QED,mol_passes_filters
    result={'Weight':None,'logP':None,'SA':None,'QED':None,'RotBond':None,'is_f':None,'pass_filter':None,'total_score':10}
    try:
        smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
        norm_mol=Chem.MolFromSmiles(smiles)

        # result['Weight']=weight(norm_mol)
        # result['logP']=logP(norm_mol)
        result['SA']=SA(norm_mol)
        result['QED']=QED(norm_mol)
        # result['RotBond']=Lipinski.NumRotatableBonds(norm_mol)
        # result['is_f']=is_flex(norm_mol)
        result['pass_filter']=mol_passes_filters(norm_mol)
    
        # result['total_score']+=10-min(max(0,result['Weight']-600)/20,10)# 0-10
        # result['total_score']+=min(10-result['logP'],7) # 0-7
        result['total_score']+=max(6-result['SA'],0) # 0-20
        result['total_score']+=result['QED'] # 0-2
        # result['total_score']+=int(not result['is_f'])*5 # 0-5
        result['total_score']+=int(result['pass_filter'])*10 # 0-10
        # result['total_score']+=10-min(10,max(0,(result['RotBond']-5))) # 0-10
    except:
        pass
    return result

def get_error(d,stat):
    if d>stat['max']:
        return (d-stat['max'])/stat['std']
    elif d<stat['min']:
        return (stat['min']-d)/stat['std']
    else:
        return 0

def scoring_func(mol,stats=None,model_type='NN'):
    result={'SMILES':None,'Weight':None,'LogP':None,'SA':None,'QED':None,'RotBond':None,'is_f':None,'pass_filter':None,'total_score':1}
    try:
    # if True:
        smiles=Chem.MolToSmiles(mol,isomericSmiles=False)
        norm_mol=Chem.MolFromSmiles(smiles)
        result['SMILES']=smiles

        result['Weight']=weight(norm_mol)
        result['LogP']=logP(norm_mol)
        result['RotBond']=Lipinski.NumRotatableBonds(norm_mol)
        result['SA']=SA(norm_mol)
        # result['SA']=sascorer.calculateScore(norm_mol)
        result['QED']=QED(norm_mol)
        result['pass_filter']=mol_passes_filters(norm_mol)
        # result['total_score']+=10*int(result['pass_filter'])
        # result['total_score']-=get_error(result['Weight'],stats['Weight'])
        # result['total_score']-=get_error(result['LogP'],stats['LogP'])
        # result['total_score']-=get_error(result['RotBond'],stats['RotBond'])
        if model_type=='NN':
            result['total_score']*=max(0.5,min(3,(5.5 - result['SA'])))
            result['total_score']*=max(0.5,min(6,10*result['QED']))
        else:
            result['total_score']*=max(0.01,min(1.5,(4 - result['SA'])))
            result['total_score']*=max(0.001,min(8,10*result['QED'])-6)
    except:
        pass
    return result


def re_rank_mol(mol,sanitize=False):
    # Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    old_conformer=mol.GetConformer()
    
    smiles=Chem.MolToSmiles(mol,isomericSmiles=False,kekuleSmiles=not sanitize)
    re_rank_core=Chem.MolFromSmiles(smiles,sanitize=sanitize)
    if not re_rank_core:
        print('molfromsmiles error in re_rank_mol!')
        return None
    
    re_rank_core.AddConformer(Chem.rdchem.Conformer(len(mol.GetAtoms())))
    re_rank_conformer=re_rank_core.GetConformer()
    match_ind=mol.GetSubstructMatch(re_rank_core)
    if match_ind:
        for i,ind in enumerate(match_ind):
            re_rank_conformer.SetAtomPosition(i,old_conformer.GetAtomPosition(ind))
        return smiles, re_rank_core
    else:
        print('GetSubstructMatch error in re_rank_mol!')
        return None

def extract_core_func(mol, selected_atoms):
    # for atom in mol.GetAtoms():
    #     atom.SetNoImplicit(True)
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)
    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom_idx in remove_atoms:
        cur_mol=new_mol.GetMol()
        atom=cur_mol.GetAtomWithIdx(atom_idx)
        for nei in atom.GetNeighbors():
            nei_idx= nei.GetIdx()
            rw_nei=new_mol.GetAtomWithIdx(nei_idx)
            if rw_nei.IsInRing():
                Bond=cur_mol.GetBondBetweenAtoms(atom_idx, nei_idx)
                rw_nei.SetNumExplicitHs(rw_nei.GetNumExplicitHs()+int(Bond.GetValenceContrib(nei)))

        new_mol.RemoveAtom(atom_idx)

    return new_mol.GetMol()

def extract_core(mol, selected_atoms):
    Chem.SanitizeMol(mol)
    core= extract_core_func(mol, selected_atoms)
    if not core:
        print('none from extract_core_func!')
        return None
    temp_mol=deepcopy(mol)
    temp_core=deepcopy(core)
    Chem.SanitizeMol(temp_mol)
    Chem.SanitizeMol(temp_core)
    if temp_mol.HasSubstructMatch(temp_core):
        rerank_result = re_rank_mol(temp_core,sanitize=True)
        if rerank_result:
            smiles, re_rank_core=rerank_result
            if temp_mol.HasSubstructMatch(re_rank_core):
                return temp_mol,smiles,re_rank_core
        return temp_mol,Chem.MolToSmiles(temp_core,isomericSmiles=False,kekuleSmiles=False),temp_core
    # print('SanitizeMol substruct matching error!')

    temp_mol=deepcopy(mol)
    temp_core=deepcopy(core)
    Chem.Kekulize(temp_mol)
    Chem.Kekulize(temp_core)
    if temp_mol.HasSubstructMatch(temp_core):
        rerank_result = re_rank_mol(temp_core,sanitize=False)
        if rerank_result:
            smiles, re_rank_core=rerank_result
            if temp_mol.HasSubstructMatch(re_rank_core):
                return temp_mol,smiles,re_rank_core
        return temp_mol,Chem.MolToSmiles(temp_core,isomericSmiles=False,kekuleSmiles=True),temp_core
    print('KekulizeMol substruct matching error!')

    if mol.HasSubstructMatch(core):
        rerank_result = re_rank_mol(core,sanitize=False)
        if rerank_result:
            smiles, re_rank_core=rerank_result
            if mol.HasSubstructMatch(re_rank_core):
                return mol,smiles,re_rank_core
        return mol,Chem.MolToSmiles(core,isomericSmiles=False,kekuleSmiles=True),core

    print('origin substruct matching error!',Chem.MolToSmiles(mol,isomericSmiles=False),Chem.MolToSmiles(core,isomericSmiles=False), Chem.MolToSmiles(re_rank_core,isomericSmiles=False))
    return None
    
# 不用
def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1: #special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls

def __extract_subgraph(mol, selected_atoms):
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots

def extract_subgraph(smiles, selected_atoms): 
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph, roots = __extract_subgraph(mol, selected_atoms) 
    try:
        # If succeeds
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
        subgraph = Chem.MolFromSmiles(subgraph)

        mol = Chem.MolFromSmiles(smiles)  # de-kekulize
        if subgraph is not None and mol.HasSubstructMatch(subgraph):
            return Chem.MolToSmiles(subgraph), roots
        else:
            return None, None
    except:
        # If fails, try without kekulization
        subgraph, roots = __extract_subgraph(mol, selected_atoms) 
        subgraph = Chem.MolToSmiles(subgraph)
        subgraph = Chem.MolFromSmiles(subgraph)
        if subgraph is not None:
            return Chem.MolToSmiles(subgraph), roots
        else:
            return None, None
