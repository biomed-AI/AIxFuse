import os
import pandas as pd
from functools import partial
from multiprocessing import Pool
from plip.structure import preparation

def get_halogen_atom_idxes(interaction,start_ind,key_res=[]):
    # print('get_halogen_atom_idxes')
    # print(interaction.halogen_bonds)
    result=[]
    res_result={}
    for ins in interaction.halogen_bonds:
        if not key_res or ins.resnr in key_res:
            if ins.resnr not in res_result:
                res_result[ins.resnr]=[]
            res_result[ins.resnr].append(ins.don_orig_idx-start_ind)
            # print(ins.don_orig_idx-start_ind)
    return res_result

def get_hbonds_atom_idxes(interaction,start_ind,key_res=[]):
    # print('get_hbonds_atom_idxes')
    # print(interaction.hbonds_ldon)
    res_result={}
    for ins in interaction.hbonds_ldon:
        if not key_res or ins.resnr in key_res:
            if ins.resnr not in res_result:
                res_result[ins.resnr]=[]
            res_result[ins.resnr].append(ins.d_orig_idx-start_ind)
            # print(ins.d_orig_idx-start_ind)
    # print(interaction.hbonds_pdon)
    for ins in interaction.hbonds_pdon:
        if not key_res or ins.resnr in key_res:
            if ins.resnr not in res_result:
                res_result[ins.resnr]=[]
            res_result[ins.resnr].append(ins.a_orig_idx-start_ind)
            # print(ins.a_orig_idx-start_ind)
    return res_result
    
def get_hydrophobic_atom_idxes(interaction,start_ind,key_res=[]):
    # print('get_hydrophobic_atom_idxes')
    # print(interaction.hydrophobic_contacts)
    res_result={}
    for ins in interaction.hydrophobic_contacts:
        if not key_res or ins.resnr in key_res:
            if ins.resnr not in res_result:
                res_result[ins.resnr]=[]
            res_result[ins.resnr].append(ins.ligatom_orig_idx-start_ind)
            # print(ins.ligatom_orig_idx-start_ind)
    return res_result

def get_pication_atom_idxes(interaction,start_ind,key_res=[]):
    # print('get_pication_atom_idxes')
    res_result={}
    # print(interaction.pication_laro)
    for ins in interaction.pication_laro:
        if not key_res or ins.charge.resnr in key_res:
            for atom_orig_idx in ins.ring.atoms_orig_idx:
                if ins.charge.resnr not in res_result:
                    res_result[ins.charge.resnr]=[]
                res_result[ins.charge.resnr].append(atom_orig_idx-start_ind)
                # print(atom_orig_idx-start_ind)
    # print(interaction.pication_paro)
    for ins in interaction.pication_paro:
        if not key_res or ins.resnr in key_res:
            for atom_orig_idx in ins.charge.atoms_orig_idx:
                if ins.resnr not in res_result:
                    res_result[ins.resnr]=[]
                res_result[ins.resnr].append(atom_orig_idx-start_ind)
                # print(atom_orig_idxstart_ind)
    return res_result

def get_pistacking_atom_idxes(interaction,start_ind,key_res=[]):
    res_result={}
    for ins in interaction.pistacking:
        if not key_res or ins.resnr in key_res:
            if ins.resnr not in res_result:
                res_result[ins.resnr]=[]
            for atom_idx in ins.ligandring.atoms_orig_idx:
                res_result[ins.resnr].append(atom_idx-start_ind)
    return res_result

def get_saltbridge_atom_idxes(interaction,start_ind,key_res=[]):
    # print('get_saltbridge_atom_idxes')
    # print(interaction.saltbridge_lneg)
    res_result={}
    for ins in interaction.saltbridge_lneg:
        if not key_res or ins.positive.resnr in key_res:
            for atom_orig_idx in ins.negative.atoms_orig_idx:
                if ins.positive.resnr not in res_result:
                    res_result[ins.positive.resnr]=[]
                res_result[ins.positive.resnr].append(atom_orig_idx-start_ind)
                # print(atom_orig_idx-start_ind)
    # print(interaction.saltbridge_pneg)
    for ins in interaction.saltbridge_pneg:
        if not key_res or ins.negative.resnr in key_res:
            for atom_orig_idx in ins.negative.atoms_orig_idx:
                if ins.negative.resnr not in res_result:
                    res_result[ins.negative.resnr]=[]
                res_result[ins.negative.resnr].append(atom_orig_idx-start_ind)
                # print(atom_orig_idx-start_ind)
    return res_result


def get_all_interaction(name,start_ind,dir,lig_identifier='UNK:Z:900'):
    my_mol = preparation.PDBComplex()
    path=os.path.join(dir,f'{name}.pdb')
    if not os.path.exists(path):
        return None
    my_mol.load_pdb(path)
    my_mol.analyze()
    interaction=my_mol.interaction_sets[lig_identifier]
    r1=get_halogen_atom_idxes(interaction,start_ind)
    r2=get_hbonds_atom_idxes(interaction,start_ind)
    r3=get_hydrophobic_atom_idxes(interaction,start_ind)
    r4=get_pication_atom_idxes(interaction,start_ind)
    r5=get_saltbridge_atom_idxes(interaction,start_ind)
    r6=get_pistacking_atom_idxes(interaction,start_ind)
    result={}
    for r in r1:
        if f'halogen_{r}' not in result:
            result[f'halogen_{r}']=0
        result[f'halogen_{r}']+=1
    for r in r2:
        if f'hbonds_{r}' not in result:
            result[f'hbonds_{r}']=0
        result[f'hbonds_{r}']+=1
    for r in r3:
        if f'hydrophobic_{r}' not in result:
            result[f'hydrophobic_{r}']=0
        result[f'hydrophobic_{r}']+=1
    for r in r4:
        if f'pication_{r}' not in result:
            result[f'pication_{r}']=0
        result[f'pication_{r}']+=1
    for r in r5:
        if f'saltbridge_{r}' not in result:
            result[f'saltbridge_{r}']=0
        result[f'saltbridge_{r}']+=1
    for r in r6:
        if 'pistacking_{r}' not in result:
            result[f'pistacking_{r}']=0
        result[f'pistacking_{r}']+=1
    
    return result
    
def get_key_inters(csv,dir,start_ind,ncpu=30):
    df=pd.read_csv(csv)
    data_list=[]
    for n in range(len(df)):
        name=df['Title'][n]
        data_list.append(name)
    work_func = partial(get_all_interaction, start_ind=start_ind,dir=dir)
    pool = Pool(ncpu)
    results = pool.map(work_func, data_list)
    final={}
    for result in results:
        if result:
            for res in result:
                if res not in final:
                    final[res]=0
                final[res]+=result[res]
    
    return final
