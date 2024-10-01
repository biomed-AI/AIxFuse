from utils_chem import extract_core_func,extract_core,expand_atoms,calc_dis
import numpy as np
from rdkit import Chem
import math
import os
import pandas as pd
from functools import partial
from multiprocessing import Pool
from Bio.PDB.PDBParser import PDBParser
from copy import deepcopy

def get_core(data,target_atoms,dis_cutoff):
    # find key atoms
    mol_with_conform,smiles,ic50=data
    try:
        old_conformer=mol_with_conform.GetConformer()
        mol=Chem.MolFromSmiles(smiles)
        mol.AddConformer(Chem.rdchem.Conformer(len(mol.GetAtoms())))
        conformer=mol.GetConformer()
        for i,atom in enumerate(mol.GetAtoms()):
            conformer.SetAtomPosition(i,old_conformer.GetAtomPosition(i))
    except:
        return None
    atom_coords=[]
    for i,atom in enumerate(mol.GetAtoms()):
        atom_coords.append(list(mol.GetConformer().GetAtomPosition(i)))
    dis_map=calc_dis(target_atoms,atom_coords)
    reserved_atoms=set([int(x) for x in np.where(np.min(dis_map,axis=0)<dis_cutoff)[0]])
    atom_list=list(reserved_atoms)
    if len(atom_list)==0:
        return None
    passed_atoms=set()
    for i in range(len(atom_list)):
        if atom_list[i] in passed_atoms:
            continue
        for j in range(len(atom_list)-1,i,-1):
            if atom_list[j] in passed_atoms:
                continue
            path = Chem.rdmolops.GetShortestPath(mol,atom_list[i],atom_list[j])
            route=set(path)-set([atom_list[i],atom_list[j]])
            for a in route:
                reserved_atoms.add(a)
                passed_atoms.add(a)
    #expand_atoms
    new_atoms=expand_atoms(set(reserved_atoms),mol)
    new_mol=Chem.RWMol(mol)
    for bond in mol.GetBonds():
        begin=bond.GetBeginAtomIdx()
        end=bond.GetEndAtomIdx()
        if begin in new_atoms or end in new_atoms:
            new_mol.RemoveBond(begin,end)
    frags=Chem.GetMolFrags(new_mol, asMols=False)
    add_atoms=set()
    for frag in frags:
        if len(frag)<4:
            add_atoms=add_atoms|set(frag)
    if add_atoms!=new_atoms:
        new_atoms=new_atoms|add_atoms

    # extract cores
    # try:
    result = extract_core(mol, new_atoms)
    if result:
        mol,smiles,re_rank_core=result
        match_ind=mol.GetSubstructMatch(re_rank_core)
        if not match_ind:
            # return mol,smiles,re_rank_core
            print('core from extract_core not sub of origin mol!',data)
            return None
        for i,ind in enumerate(match_ind):
            atom = mol.GetAtomWithIdx(ind)
            for nei in atom.GetNeighbors():
                nei_ind=nei.GetIdx()
                if nei_ind not in match_ind:
                    atom.SetAtomMapNum(i+1)
                    nei.SetAtomMapNum(i+1)
        reserved_atoms = set([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in new_atoms])
        Chem.SanitizeMol(mol)
        new_mol=extract_core_func(mol, reserved_atoms)
        # Chem.Kekulize(mol)
        # Chem.Kekulize(new_mol)
        Chem.SanitizeMol(mol)
        Chem.SanitizeMol(new_mol)
        Chem.SanitizeMol(re_rank_core)
        frags=Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
        return mol,smiles,re_rank_core,frags,ic50
    else:
        print('none from extract_core!',data)
        return None
    # except:
    #     print(2,data,key_inter,start_ind)
    #     return None


def get_cores(key_res,csv,prot_pdb,mol_sdf,dis_cutoff=4,ncpu=30):
    df=pd.read_csv(csv)
    data_info={}
    for n in range(len(df)):
        name=df['Title'][n]
        smiles=df['SMILES'][n]
        ic50=df['IC50nM'][n]
        if not ic50 or np.isnan(ic50):
            ic50=0.001
        ic50=-9+math.log10(ic50)
        data_info[name]=(smiles,ic50)

    p = PDBParser(PERMISSIVE=1)
    chain =p.get_structure('', prot_pdb)[0]['A']
    for res_id in str(key_res).split(','):
        residue=chain[int(res_id)]
        target_atoms=[]
        for atom in residue.get_list():
            target_atoms.append(atom.get_coord())

    
    data_list=[]
    mol_list=Chem.SDMolSupplier(mol_sdf)
    
    for mol in mol_list:
        title=mol.GetProp('s_m_title')
        if title in data_info:
            smiles,ic50=data_info[title]
            data_list.append((mol,smiles,ic50))
        
    work_func = partial(get_core, target_atoms=target_atoms,dis_cutoff=dis_cutoff)
    pool = Pool(ncpu)
    results = pool.map(work_func, data_list)
    smiles_dict={}
    for i,result in enumerate(results):
        if result is not None:
            mol_with_conform,smiles,core,frags,ic50 = result
            if mol_with_conform and core:
                if smiles not in smiles_dict:
                    smiles_dict[smiles]={}
                    smiles_dict[smiles]['act_mols']=[]
                    smiles_dict[smiles]['core']=core
                    smiles_dict[smiles]['iec50']=[]
                smiles_dict[smiles]['act_mols'].append(mol_with_conform)
                smiles_dict[smiles]['iec50'].append(ic50)
                if len(frags)==0:
                    print(1,smiles)
                for frag in frags:
                    for atom in frag.GetAtoms():
                        site = atom.GetAtomMapNum()-1
                        if site>-1:
                            smiles_dict[smiles]['core'].GetAtomWithIdx(site).SetAtomMapNum(site+1)
                            if site not in smiles_dict[smiles]:
                                smiles_dict[smiles][site]=[]
                            smiles_dict[smiles][site].append(frag)
                            break
    return smiles_dict

