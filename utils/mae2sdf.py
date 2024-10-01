from schrodinger import structure
import schrodinger.thirdparty.rdkit_adapter as rdkit_adapter
import pickle
import argparse
from rdkit import Chem
from tqdm import tqdm

def parse_mae(lig_mae,mol_sdf,pK_thrh):
    reader = structure.StructureReader(lig_mae)
    writer=Chem.SDWriter(mol_sdf)
    title_dict={}
    for st in tqdm(reader):
        if 'r_i_docking_score' in st.property:
            score= float(st.property["r_i_docking_score"])
            if score > pK_thrh*-1.36357:
                continue
            mol=rdkit_adapter.to_rdkit(st)
            mol=Chem.rdmolops.RemoveAllHs(mol)
            title=str(st.property["s_m_title"])
            mol.SetProp('s_m_title',title)
            mol.SetProp('r_i_docking_score',str(st.property["r_i_docking_score"]))
            if title not in title_dict or score < float(title_dict[title].GetProp('r_i_docking_score')):
                title_dict[title]=mol
    for title in title_dict:
        writer.write(title_dict[title])
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lig_mae')
    parser.add_argument('--mol_sdf')
    parser.add_argument('--pK_thrh',type=int,default=0)
    args = parser.parse_args()
    parse_mae(args.lig_mae,args.mol_sdf,args.pK_thrh)