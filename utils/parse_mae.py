from schrodinger import structure
import schrodinger.thirdparty.rdkit_adapter as rdkit_adapter
import pickle
import argparse
from rdkit import Chem
from tqdm import tqdm
from glob import glob
import os
def parse_mae(args):
    for lig_mae in glob(os.path.join(args.dir,'*.maegz')):
        mol_sdf=lig_mae[:-5]+'sdf'
        name = lig_mae.split('/')[-1].split('.')[0]
        # if name not in ['AIxFuse','MARS','ReINVENT','RationaleRL','jnk3_act','gsk3b_act','rorgt_act','dhodh_act','JMC']:
        #     continue
        if name not in ['AIxFuse_init']:
            continue
        reader = structure.StructureReader(lig_mae)
        mol_dict={}
        for st in tqdm(reader):
            if 'r_i_docking_score' in st.property:
                mol=rdkit_adapter.to_rdkit(st)
                # mol=Chem.rdmolops.RemoveAllHs(mol)
                title = str(st.property["s_m_title"])
                score = float(st.property["r_i_docking_score"])
                if title not in mol_dict or score < mol_dict[title][0]:
                    mol_dict[title]=(score,mol)
        writer=Chem.SDWriter(mol_sdf)
        for title in mol_dict:
            mol=mol_dict[title][1]
            writer.write(mol)
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    args = parser.parse_args()
    parse_mae(args)