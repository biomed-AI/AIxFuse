import os
from plip.structure import preparation
from rdkit import Chem
from utils_chem import expand_atoms,extract_core,extract_core_func,calc_dis
import numpy as np
import pandas as pd
import math
from functools import partial
from multiprocessing import Pool
from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem

def get_core_info(mol,contact_atoms,log_ic50,max_r_group_atom=4):
    atom_list=list(contact_atoms)
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
                contact_atoms.add(a)
                passed_atoms.add(a)
    new_atoms=expand_atoms(set(contact_atoms),mol)
    new_mol=Chem.RWMol(mol)
    for bond in mol.GetBonds():
        begin=bond.GetBeginAtomIdx()
        end=bond.GetEndAtomIdx()
        if begin in new_atoms or end in new_atoms:
            new_mol.RemoveBond(begin,end)
    frags=Chem.GetMolFrags(new_mol, asMols=False)
    add_atoms=set()
    for frag in frags:
        if len(frag)<=max_r_group_atom:
            add_atoms=add_atoms|set(frag)
    if add_atoms!=new_atoms:
        new_atoms=new_atoms|add_atoms
    result = extract_core(mol, new_atoms)
    if result:
        mol,smiles,re_rank_core=result
        new_mol=Chem.RWMol(mol)
        for bond in mol.GetBonds():
            begin_atom=bond.GetBeginAtomIdx()
            end_atom=bond.GetEndAtomIdx()
            if begin_atom not in new_atoms and end_atom not in new_atoms:
                new_mol.RemoveBond(begin_atom,end_atom)
        
        match_ind=new_mol.GetSubstructMatch(re_rank_core)
        if not match_ind:
            # return mol,smiles,re_rank_core
            return None
        for i,ind in enumerate(match_ind):
            atom = mol.GetAtomWithIdx(ind)
            for nei in atom.GetNeighbors():
                nei_ind=nei.GetIdx()
                if nei_ind not in match_ind:
                    atom.SetAtomMapNum(i+1)
                    nei.SetAtomMapNum(i+1)
        contact_atoms = set([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in new_atoms])
        Chem.SanitizeMol(mol)
        new_mol=extract_core_func(mol, contact_atoms)
        # Chem.Kekulize(mol)
        # Chem.Kekulize(new_mol)
        Chem.SanitizeMol(mol)
        Chem.SanitizeMol(new_mol)
        Chem.SanitizeMol(re_rank_core)
        frags=Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
        return mol,smiles,re_rank_core,frags,log_ic50,new_atoms
    else:
        return None

def get_inter_res(inter_list):
    inter_res=set()
    for inter in inter_list:
        inter_res.add(f'{inter[0]}_{inter[3]}')
    return inter_res

def get_score_weight(atom_num,max_frag_atom_num):
    if max_frag_atom_num < atom_num/6 or max_frag_atom_num > 5*atom_num/6:
        return 0
    if max_frag_atom_num < atom_num/3:
        return max_frag_atom_num*3/atom_num
    if max_frag_atom_num > 2*atom_num/3:
        return 5 - 6*max_frag_atom_num/atom_num
    return 1

def get_info(data,start_ind,dir,conformer_dir,lig_identifier='UNK:Z:900',contact_res=set(),reject_res=set(),reserved_inter=set(),score_thrh=None,dis_thrh=4):
    name,smiles,log_ic50=data
    result=[]
    
    my_mol = preparation.PDBComplex()
    path=os.path.join(dir,f'{name}.pdb')
    conformer_pdb=os.path.join(conformer_dir,f'{name}.pdb')
    if not os.path.exists(path):
        return None
    try:
        mol=Chem.MolFromSmiles(smiles)
        mol_with_conform=Chem.MolFromPDBFile(conformer_pdb)
        old_conformer=mol_with_conform.GetConformer()
        mol.AddConformer(Chem.rdchem.Conformer(len(mol.GetAtoms())))
        conformer=mol.GetConformer()
        for i,atom in enumerate(mol.GetAtoms()):
            conformer.SetAtomPosition(i,old_conformer.GetAtomPosition(i))
        mol_with_conform=mol
    except:
        return None
    score_list=[0 for _ in range(len(mol.GetAtoms()))]
    my_mol.load_pdb(path)
    my_mol.analyze()
    interaction=my_mol.interaction_sets[lig_identifier]

    p = PDBParser(PERMISSIVE=1)
    comp=p.get_structure('', path)[0]
    chain =comp['A']
    
            
    ligand=comp.get_list()[1]
    ligand_atoms=[]
    for residue in ligand.get_list():
        for atom in residue.get_list():
            ligand_atoms.append(atom.get_coord())
    ligand_atoms=ligand_atoms[:len(mol.GetAtoms())]

    if contact_res:
        target_atoms_list=[]
        for res_id in contact_res:
            target_atoms=[]
            residue=chain[int(res_id)]
            for atom in residue.get_list():
                target_atoms.append(atom.get_coord())
                target_atoms_list.append(atom.get_coord())
            dis_map=calc_dis(ligand_atoms,target_atoms)
            min_dis_count=np.min(dis_map,axis=1)
            if np.min(min_dis_count)<=6:
                result.append(('contact','min',2,res_id,[int(np.argmin(min_dis_count))]))
        dis_map=calc_dis(ligand_atoms,target_atoms_list)
        min_dis_count=np.min(dis_map,axis=1)
        score_list+=(min_dis_count<3)*1+(min_dis_count>=3)*(min_dis_count<=6)*(min_dis_count-6)/-3
        # if not result:
        #     return None
    
    if reject_res:
        off_target_atoms_list=[]
        for res_id in reject_res:
            residue=chain[int(res_id)]
            for atom in residue.get_list():
                off_target_atoms_list.append(atom.get_coord())
        dis_map=calc_dis(ligand_atoms,off_target_atoms_list)
        min_dis_rej=np.min(dis_map,axis=1)
        score_list-=(min_dis_rej<3)*1+(min_dis_rej>=3)*(min_dis_rej<=6)*(min_dis_rej-6)/-3
    
    # hbond
    for ins in interaction.hbonds_ldon:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        result.append(('hbond','ldon',2,ins.resnr,[ins.d_orig_idx-start_ind]))
        score_list[ins.d_orig_idx-start_ind]+=2
    for ins in interaction.hbonds_pdon:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        result.append(('hbond','pdon',2,ins.resnr,[ins.a_orig_idx-start_ind]))
        score_list[ins.a_orig_idx-start_ind]+=2
    # hydrophobic
    for ins in interaction.hydrophobic_contacts:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        result.append(('hydrophobic','',1,ins.resnr,[ins.ligatom_orig_idx-start_ind]))
        score_list[ins.ligatom_orig_idx-start_ind]+=1
    # pistacking
    for ins in interaction.pistacking:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        atoms=[]
        for atom_idx in ins.ligandring.atoms_orig_idx:
            atoms.append(atom_idx-start_ind)
            score_list[atom_idx-start_ind]+=1/len(ins.ligandring.atoms_orig_idx)
        result.append(('pistacking','',1,ins.resnr,atoms))
    # pication
    for ins in interaction.pication_laro:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        atoms=[]
        for atom_idx in ins.ring.atoms_orig_idx:
            atoms.append(atom_idx-start_ind)
            score_list[atom_idx-start_ind]+=1/len(ins.ring.atoms_orig_idx)
        result.append(('pication','laro',1,ins.resnr,atoms))
    for ins in interaction.pication_paro:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        atoms=[]
        for atom_idx in ins.charge.atoms_orig_idx:
            atoms.append(atom_idx-start_ind)
            score_list[atom_idx-start_ind]+=1/len(ins.charge.atoms_orig_idx)
        result.append(('pication','paro',1,ins.resnr,atoms))
    # saltbridge
    for ins in interaction.saltbridge_lneg:
        # if reject_res and ins.positive.resnr in reject_res:
        #     continue
        atoms=[]
        for atom_idx in ins.negative.atoms_orig_idx:
            atoms.append(atom_idx-start_ind)
            score_list[atom_idx-start_ind]+=1/len(ins.negative.atoms_orig_idx)
        result.append(('saltbridge','lneg',1,ins.positive.resnr,atoms))
    for ins in interaction.saltbridge_pneg:
        # if reject_res and ins.negative.resnr in reject_res:
        #     continue
        atoms=[]
        for atom_idx in ins.positive.atoms_orig_idx:
            atoms.append(atom_idx-start_ind)
            score_list[atom_idx-start_ind]+=1/len(ins.positive.atoms_orig_idx)
        result.append(('saltbridge','pneg',1,ins.negative.resnr,atoms))
    # halogen
    for ins in interaction.halogen_bonds:
        # if reject_res and ins.resnr in reject_res:
        #     continue
        result.append(('halogen','',1,ins.resnr,[ins.don_orig_idx-start_ind]))
        score_list[ins.don_orig_idx-start_ind]+=1
    for i in range(len(mol.GetAtoms())):
        score_list[i]=max(-2,min(2,score_list[i]))

    atom_num=len(mol.GetAtoms())
    final_result={}
    for inter1 in result:
        if reserved_inter and inter1[0] not in reserved_inter:
            continue
        atom_list=inter1[-1]
        inter_res=get_inter_res([inter1])

        contact_atoms=set(atom_list)
        core_info=get_core_info(mol,contact_atoms,log_ic50)
        if not core_info:
            continue
        sub_atoms=core_info[-1]
        smiles=core_info[1]
        if len(sub_atoms)<=6:
            continue
        # if contact_res and not contact_atoms.intersection(sub_atoms):
        #     continue
        # if reject_res and reject_atoms.intersection(sub_atoms):
        #     continue
        score=0
        for atom in sub_atoms:
            score+=score_list[atom]
        frags=core_info[3]
        max_frag_atom_num=0 if not frags else max([len(frag.GetAtoms()) for frag in frags])
        score=score*get_score_weight(atom_num,max_frag_atom_num)
        # *(max(0,min(5,len(sub_atoms)-5))-max(0,5*min(1,2*(len(sub_atoms)-atom_num//2)/(atom_num-atom_num//2))))
        sub_atoms_list=list(core_info[-1])
        sub_atoms_list.sort()
        key = tuple(set(sub_atoms_list))
        if key not in final_result or score>final_result[key][1]:
            final_result[key]=(name,score,core_info,inter_res,sub_atoms)
    
    for i,inter1 in enumerate(result):
        for j,inter2 in enumerate(result[i+1:]):
            if reserved_inter and not reserved_inter.intersection(set([inter1[0],inter2[0]])):
                continue
            atom_list=inter1[-1]+inter2[-1]
            inter_res=get_inter_res([inter1,inter2])

            contact_atoms=set(atom_list)
            core_info=get_core_info(mol,contact_atoms,log_ic50)
            if not core_info:
                continue
            sub_atoms=core_info[-1]
            smiles=core_info[1]
            if len(sub_atoms)<=6:
                continue
            # if contact_res and not contact_atoms.intersection(sub_atoms):
            #     continue
            # if reject_res and reject_atoms.intersection(sub_atoms):
            #     continue
            score=0
            for atom in sub_atoms:
                score+=score_list[atom]
            frags=core_info[3]
            max_frag_atom_num=0 if not frags else max([len(frag.GetAtoms()) for frag in frags])
            score=score*get_score_weight(atom_num,max_frag_atom_num)
            sub_atoms_list=list(core_info[-1])
            sub_atoms_list.sort()
            key = tuple(set(sub_atoms_list))
            if key not in final_result or score>final_result[key][1]:
                final_result[key]=(name,score,core_info,inter_res,sub_atoms)
    for i,inter1 in enumerate(result):
        for j,inter2 in enumerate(result[i+1:]):
            for inter3 in result[i+1+j+1:]:
                if reserved_inter and not reserved_inter.intersection(set([inter1[0],inter2[0],inter3[0]])):
                    continue
                atom_list=inter1[-1]+inter2[-1]+inter3[-1]
                inter_res=get_inter_res([inter1,inter2,inter3])
                contact_atoms=set(atom_list)
                core_info=get_core_info(mol,contact_atoms,log_ic50)
                if not core_info:
                    continue
                sub_atoms=core_info[-1]
                smiles=core_info[1]
                if len(sub_atoms)<=6:
                    continue
                # if contact_res and not contact_atoms.intersection(sub_atoms):
                #     continue
                # if reject_res and reject_atoms.intersection(sub_atoms):
                #     continue
                score=0
                for atom in sub_atoms:
                    score+=score_list[atom]
                frags=core_info[3]
                max_frag_atom_num=0 if not frags else max([len(frag.GetAtoms()) for frag in frags])
                score=score*get_score_weight(atom_num,max_frag_atom_num)
                sub_atoms_list=list(core_info[-1])
                sub_atoms_list.sort()
                key = tuple(set(sub_atoms_list))
                if key not in final_result or score>final_result[key][1]:
                    final_result[key]=(name,score,core_info,inter_res,sub_atoms)
    # for i,inter1 in enumerate(result):
    #     for j,inter2 in enumerate(result[i+1:]):
    #         for k,inter3 in enumerate(result[i+1+j+1:]):
    #             for inter4 in result[i+1+j+1+k+1:]:
    #                 if reserved_inter and not reserved_inter.intersection(set([inter1[0],inter2[0],inter3[0],inter4[0]])):
    #                     continue
    #                 atom_list=inter1[-1]+inter2[-1]+inter3[-1]+inter4[-1]
    #                 total_weight=inter1[2]+inter2[2]+inter3[2]+inter4[2]
    #                 inter_res=get_inter_res([inter1,inter2,inter3,inter4])
    #                 contact_atoms=set(atom_list)
    #                 core_info=get_core_info(mol,contact_atoms,log_ic50)
    #                 if not core_info:
    #                     continue
    #                 sub_atoms=core_info[-1]
    #                 if contact_res and not contact_atoms.intersection(sub_atoms):
    #                     continue
    #                 if reject_res and reject_atoms.intersection(sub_atoms):
    #                     continue
    #                 sub_atoms_list=list(core_info[-1])
    #                 sub_atoms_list.sort()
    #                 key = tuple(set(sub_atoms_list))
    #                 score=total_weight*(max(1,min(6,len(sub_atoms)-4))-max(0,min(5,len(sub_atoms)-15)))
    #                 if key not in final_result or score>final_result[key][1]:
    #                     final_result[key]=(name,score,core_info,inter_res,sub_atoms)

    if not final_result:
        return None 
    key_list=[]
    score_list=[]
    for key in final_result:
        score=final_result[key][1]
        key_list.append(key)
        score_list.append(score)
    
    result=[]
    for key in [key_list[i] for i in np.argsort(-np.array(score_list))]:
        if score_thrh is not None:
            score=final_result[key][1]
            if score >=score_thrh:
                result.append(final_result[key])
        else:
            result.append(final_result[key])
    for res in result:
        print(res)
    return result


def get_infos(csv,start_ind,dir,conformer_dir,lig_identifier='UNK:Z:900',contact_res=set(),reject_res=set(),reserved_inter=set(),score_thrh=None,ncpu=30):
    df=pd.read_csv(csv)
    data_list=[]
    work_func = partial(get_info, start_ind=start_ind,dir=dir,conformer_dir=conformer_dir,lig_identifier=lig_identifier,contact_res=contact_res,reject_res=reject_res,reserved_inter=reserved_inter,score_thrh=score_thrh)
    for n in range(len(df)):
        name=df['Title'][n]
        smiles=df['SMILES'][n]
        ic50=df['IC50nM'][n]
        if not ic50 or np.isnan(ic50):
            ic50=0.001
        log_ic50=-9+math.log10(ic50)
        data_list.append((name,smiles,log_ic50))
    pool = Pool(ncpu)
    results = pool.map(work_func, data_list)
    score_dict={}
    for i,result in enumerate(results):
        if result is not None:
            for name,score,core_info,inter_res,sub_atoms in result:
                mol_with_conform,smiles,core,frags,log_ic50,new_atoms=core_info
                if mol_with_conform and core:
                    if smiles not in score_dict:
                        score_dict[smiles]={}
                        score_dict[smiles]['scores']=[]
                        score_dict[smiles]['name']=[]
                        score_dict[smiles]['sub_atoms']=[]
                    score_dict[smiles]['scores'].append(score)
                    score_dict[smiles]['name'].append(name)
                    score_dict[smiles]['sub_atoms'].append(sub_atoms)
    smiles_list=[]
    count_list=[]
    for smiles in score_dict:
        smiles_list.append(smiles)
        count_list.append(np.sum(score_dict[smiles]['scores']))
    arg_idx=np.argsort(count_list)[::-1]
    smiles_dict={}
    name_dict={}
    for smiles in [smiles_list[i] for i in arg_idx]:
        smiles_dict[smiles]=[]
        for name,sub_atoms in zip(score_dict[smiles]['name'],score_dict[smiles]['sub_atoms']):
            if name not in name_dict:
                name_dict[name]=set(sub_atoms)
                smiles_dict[smiles].append(name)
            elif not name_dict[name].intersection(sub_atoms):
                name_dict[name]=name_dict[name].union(set(sub_atoms))
                smiles_dict[smiles].append(name)


    smiles_info={}
    for i,result in enumerate(results):
        if result is not None:
            for name,score,core_info,inter_res,sub_atoms in result:
                mol_with_conform,smiles,core,frags,log_ic50,new_atoms=core_info
                if name not in smiles_dict[smiles]:
                    continue
                if mol_with_conform and core:
                    if smiles not in smiles_info:
                        smiles_info[smiles]={}
                        smiles_info[smiles]['act_mols']=[]
                        smiles_info[smiles]['name']=[]
                        smiles_info[smiles]['core']=core
                        smiles_info[smiles]['iec50']=[]
                        smiles_info[smiles]['scores']=[]
                    smiles_info[smiles]['scores'].append(score)
                    smiles_info[smiles]['name'].append(name)
                    smiles_info[smiles]['act_mols'].append(mol_with_conform)
                    smiles_info[smiles]['iec50'].append(log_ic50)
                    if len(frags)==0:
                        print('No side chains:',smiles)
                    for frag in frags:
                        for atom in frag.GetAtoms():
                            site = atom.GetAtomMapNum()-1
                            if site>-1:
                                smiles_info[smiles]['core'].GetAtomWithIdx(site).SetAtomMapNum(site+1)
                                if site not in smiles_info[smiles]:
                                    smiles_info[smiles][site]=[]
                                smiles_info[smiles][site].append(frag)
                                break
    return smiles_info