import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os
import glob

from rdkit.Chem import AllChem,Descriptors,Lipinski,DataStructs
import numpy as np
def dedup(raw_csv,csv):
    df_raw=pd.read_csv(raw_csv)
    smiles_dict={}
    for smiles,title,ic50nM in tqdm(zip(df_raw['SMILES'],df_raw['Title'],df_raw['IC50nM'])):
        try:
            norm_smiles=Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=False)
        except:
            continue
        try:
            ic50nM_value=999999999
            ic50nM_value=float(ic50nM)
        except:
            if ic50nM[0]=='<':
                ic50nM_value=float(ic50nM[1:])
        if ic50nM_value>=1000:
            continue
        if norm_smiles not in smiles_dict or smiles_dict[norm_smiles][0]>ic50nM_value:
            
                smiles_dict[norm_smiles]=(ic50nM_value,title)
    
    smiles_list=[]
    title_list=[]
    ic50nM_list=[]
    for norm_smiles in smiles_dict:
        smiles_list.append(norm_smiles)
        ic50nM,title=smiles_dict[norm_smiles]
        title_list.append(title)
        ic50nM_list.append(ic50nM)
    df=pd.DataFrame({'SMILES':smiles_list,'Title':title_list,'IC50nM':ic50nM_list})
    df.to_csv(csv,index=False)

def merge_pdbs(target_pdb,act_pdbs,merged_pdbs):
    
    target_lines=open(target_pdb).readlines()
    for i,line in enumerate(target_lines[::-1]):
        if line.startswith('A') or line.startswith('H'):
            break
    sep_line=len(target_lines)-i
    print(sep_line)
    if not os.path.exists(merged_pdbs):
        os.mkdir(merged_pdbs)
    for file in glob.glob(act_pdbs+'/*'):
        lines=open(file).readlines()
        this_lines=[]
        for line in lines:
            if len(line)>6 and line[:6]=='HETATM':
                this_lines.append(line)
        merged_lines=target_lines[:sep_line]+this_lines+target_lines[sep_line:]
        name=file.split('/')[-1]
        file=os.path.join(merged_pdbs,name)
        with open(file,'w') as w:
            for line in merged_lines:
                w.write(line)

def get_properties(df,ncpu=64):
    from moses.metrics import weight, logP, SA, QED,mol_passes_filters
    from moses.metrics.utils import get_mol, mapper
    mols=mapper(ncpu)(get_mol,df['SMILES'])
    print('got mols')
    tmp_list=[]
    ind_list=[]
    for i,mol in enumerate(mols):
        if mol is not None:
            tmp_list.append(mol)
            ind_list.append(i)
    mols = tmp_list
    df['SA']=[None for _ in range(len(df))]
    df['SA'].iloc[ind_list]=mapper(32)(SA,mols)
    print('got sa')
    df['QED']=[None for _ in range(len(df))]
    df['QED'].iloc[ind_list]=mapper(32)(QED,mols)
    print('got qed')
    df['Weight']=[None for _ in range(len(df))]
    df['Weight'].iloc[ind_list]=mapper(32)(weight,mols)
    print('got weight')
    df['LogP']=[None for _ in range(len(df))]
    df['LogP'].iloc[ind_list]=mapper(32)(logP,mols)
    print('got logp')
    df['TPSA']=[None for _ in range(len(df))]
    df['TPSA'].iloc[ind_list]=mapper(1)(Descriptors.TPSA,mols)
    print('got tpsa')
    df['HBA']=[None for _ in range(len(df))]
    df['HBA'].iloc[ind_list]=mapper(1)(Lipinski.NumHAcceptors,mols)
    print('got hba')
    df['HBD']=[None for _ in range(len(df))]
    df['HBD'].iloc[ind_list]=mapper(1)(Lipinski.NumHDonors,mols)
    print('got hbd')
    df['RotBond']=[None for _ in range(len(df))]
    df['RotBond'].iloc[ind_list]=mapper(1)(Lipinski.NumRotatableBonds,mols)
    print('got rotbond')
    df['Filtered']=[None for _ in range(len(df))]
    df['Filtered'].iloc[ind_list]=mapper(32)(mol_passes_filters,mols)
    return df

def get_free_node(device='gpu'):
    ncpu=92 if device=='gpu' else 30
    lines=os.popen('ssh 192.168.2.103 \'pestat\'')
    node_partion_list=[]
    free_cpu_list=[]
    for line in lines:
        s=line.split()
        if s[0].startswith(device):
            node,partion,state,cpu_use,cpu_total,cpu_load,memsize,freemem=s[:8]
            node_partion_list.append((node,''.join(partion.split('*')),ncpu))
            free_cpu_list.append(int(cpu_total)-int(cpu_use))
    max_ind=np.argmax(free_cpu_list)
    return node_partion_list[max_ind]
