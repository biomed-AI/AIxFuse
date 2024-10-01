import pandas as pd
import sys
import argparse
from tqdm import tqdm


def mae_to_csv(inputfile,outputfile):
    from schrodinger import structure
    count=0
    title_list=[]
    score_list=[]
    with structure.StructureReader(inputfile) as loaded_st:
        for st in tqdm(loaded_st):
            count = count+1
            try:
                title=st.property["s_m_title"]
                score=st.property["r_i_docking_score"]
                title_list.append(title)
                score_list.append(score)
            except:
                pass

    df = pd.DataFrame({'Title':title_list,'docking score':score_list})
    df = df[['Title','docking score']]
    df = df.groupby('Title').agg('min')
    df.to_csv(outputfile)

def mae_split(inputfile,inputcsv,outputfile):
    from schrodinger import structure
    df=pd.read_csv(inputcsv)
    title_set=set(df['Title'])

    writer = structure.StructureWriter(outputfile)
    with structure.StructureReader(inputfile) as loaded_st:
        for st in tqdm(loaded_st):
            try:
                title=st.property["s_m_title"]
                if title in title_set:
                    writer.append(st)
            except:
                pass
    writer.close()

def merge_csvs(inputs,output,threshold=-7.5):
    inputs=inputs.split(',')
    scores={}
    names=[]
    for input in inputs:
        name = input.split('/')[-1].split('.')[0]
        names.append(name)
    for input in inputs:
        name = input.split('/')[-1].split('.')[0]
        df=pd.read_csv(input)
        for title,score in zip(df['Title'],df['docking score']):
            if title not in scores:
                scores[title]={}
                for n in names:
                    scores[title][n]=None
            scores[title][name]=score
    
    title_list=list(scores.keys())
    df2_dict={'SMILES':[None for _ in title_list],'Title':title_list}
    ind_list=None
    for name in names:
        score_list=[]
        for title in title_list:
            score_list.append(scores[title][name])
        df2_dict[name]=score_list
    df2=pd.DataFrame(df2_dict)
    for name in names:
        if ind_list is None:
            ind_list=df2[name]<threshold
        else:
            ind_list=ind_list&(df2[name]<threshold)
    df3=df2[ind_list]
    df3.to_csv(output,index=False)

def get_scores(info_csv,maes,output_csv):
    df=pd.read_csv(info_csv)
    if 'SMILES' not in df:
        df=pd.read_csv(info_csv,'\t')
    if 'SMILES' not in df:
        df=pd.read_csv(info_csv,' ')
    from schrodinger import structure
    titles=set([str(t).strip() for t in df['Title']])
    print(len(titles))
    for mae in maes.split(','):
        score_dict={}
        with structure.StructureReader(mae) as loaded_st:
            for st in tqdm(loaded_st):
                title=st.property["s_m_title"].strip()
                if title in titles:
                    score=st.property["r_i_docking_score"]
                    if title in score_dict:
                        score_dict[title]=min(score,score_dict[title])
                    else:
                        score_dict[title]=score
    
        score_list=[None for _ in range(len(df))]
        for i,title in enumerate(df['Title']):
            if str(title) in score_dict:
                score_list[i]=score_dict[str(title)]
        target=mae.split('/')[-2]
        df[target]=score_list
    df.to_csv(output_csv,index=False)



def get_smiles(input,name_input,output):
    df=pd.read_csv(name_input)
    smiles_dict={}
    for title,smiles,sa in zip(df['Title'],df['SMILES'],df['SA']):
        smiles_dict[title]=(smiles,sa)
    smiles_list=[]
    df2=pd.read_csv(input)
    sa_list=[]
    for title in df2['Title']:
        smiles,sa=smiles_dict[title]
        smiles_list.append(smiles)
        sa_list.append(sa)
    df2['SMILES']=smiles_list
    df2['SA']=sa_list
    df2.to_csv(output,index=False)

def split_csv(input,output):
    import numpy as np
    from collections import OrderedDict
    from rdkit import Chem,DataStructs
    from rdkit.Chem import AllChem
    from tqdm import trange
    df=pd.read_csv(input)
    names=set(df.columns)-set(['Title','SMILES'])
    df['total_score']=[0 for _ in range(len(df))]
    for name in names:
        df['total_score']=df['total_score']+df[name]
    ind_list=np.argsort(df['total_score'])
    df=df.iloc[ind_list]
    fp_dict=OrderedDict()
    for i in trange(len(df)):
        smiles=df['SMILES'][i]
        fp=AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2,1024)
        flag=True
        for fp2 in fp_dict:
            if DataStructs.TanimotoSimilarity(fp,fp2)>0.5:
                fp_dict[fp2].append(i)
                flag=False
                break
        if flag:
            fp_dict[fp]=[i]
    
    for i,fp in enumerate(fp_dict):
        df2=df.iloc[fp_dict[fp]]
        df2.to_csv('{}/sep_{:>4d}.csv'.format(output,i),index=False)

def sep_mae(input_maes,input_csvs,output):
    from schrodinger import structure
    from glob import glob
    
    mae_names = input_maes.split(',')
    st_dict_dict={}
    for mae_name in mae_names:
        target_name = mae_name.split('/')[-2]
        st_dict={}
        with structure.StructureReader(mae_name) as loaded_st:
            for st in tqdm(loaded_st):
                try:
                    title=st.property["s_m_title"]
                    score=st.property["r_i_docking_score"]
                    if title not in st_dict:
                        st_dict[title]=st
                    elif score < st_dict[title].property["r_i_docking_score"]:
                        st_dict[title]=st
                except:
                    pass
        st_dict_dict[target_name]=st_dict
    for target_name in st_dict_dict:
        st_dict=st_dict_dict[target_name]
        for input_csv in glob(input_csvs+'/*'):
            df=pd.read_csv(input_csv)
            file_name=input_csv.split('/')[-1].split('.')[0]
            outputfile='{}/{}_{}.mae'.format(output,target_name,file_name)
            writer = structure.StructureWriter(outputfile)
            for title in df['Title']:
                st=st_dict[title]
                writer.append(st)
            writer.close()

def merge_properties(input,info,output):
    from moses.metrics.utils import get_mol, mapper
    from rdkit.Chem import Descriptors,AllChem
    import numpy as np
    from rdkit.Chem import Draw
    from rdkit import Chem
    df1=pd.read_csv(input)
    df2=pd.read_csv(info)
    title_dict={}
    for title, total_pk, pK_JAK1,pK_PRK in zip(df1['Title'],df1['Total_pK'],df1['JAK1_pK'],df1['PRKAB1_pK']):
        title_dict[title]=(total_pk, pK_JAK1,pK_PRK)

    ind_list=[]
    total_pks=[]
    jak1_pks=[]
    prk_pks=[]
    for ind, title in enumerate(df2['Title']):
        if title in title_dict:
            total_pk, pK_JAK1,pK_PRK=title_dict[title]
            ind_list.append(ind)
            total_pks.append(total_pk)
            jak1_pks.append(pK_JAK1)
            prk_pks.append(pK_PRK)
    
    df3=df2.iloc[ind_list]
    df3['Total_pK']=total_pks
    df3['JAK1_pK']=jak1_pks
    df3['PRKAB1_pK']=prk_pks
    mols = mapper(32)(get_mol, df3['SMILES'])
    ha_num=mapper(1)(Descriptors.HeavyAtomCount, mols)
    df3['JAK1_LE']=df3['JAK1_pK']*1.36357/ha_num
    df3['PRKAB1_LE']=df3['PRKAB1_pK']*1.36357/ha_num

    pk_score1=df3['JAK1_pK']-6#docking分数
    pk_score2=df3['PRKAB1_pK']-6.7
    le_score1=df3['JAK1_LE']#配体效率
    le_score2=df3['PRKAB1_LE']

    sa_score=3-df3['SA']#合成可及性
    rotbond_score=7-df3['RotBond']#可旋转键数量
    rotbond_score[rotbond_score>1]=1#截断

    qed_score=df3['QED']#类药性
    logp_score=5-df3['LogP']#脂水分配系数
    logp_score[logp_score>1]=1
    hba_score=11-df3['HBA']#氢键受体数量
    hba_score[hba_score>1]=1
    hbd_score=6-df3['HBD']#氢键供体数量
    hbd_score[hbd_score>1]=1

    df3['Score']=\
        2*(pk_score1+le_score1)+\
        3*(pk_score2+le_score2)+\
        6*sa_score+\
        1*rotbond_score+\
        (qed_score+logp_score+hba_score+hbd_score)
    
    df3=df3[df3['JAK1_pK']>=6]
    df3=df3[df3['PRKAB1_pK']>=6.7]
    df3=df3[df3['Total_pK']>=13.5]
    df3=df3[df3['SA']<=3.5]
    sort_ix=np.argsort(-df3['Score'])
    df3=df3.iloc[sort_ix]
    mols=[Chem.MolFromSmiles(s) for s in df3['SMILES']]

    df3.to_csv(output,index=False)
    print(mols)
    img=Draw.MolsToGridImage(mols[:200], molsPerRow=5, subImgSize=(600, 600),maxMols=400,returnPNG=True)
    img.save('/home/chensheng/projects/AIXMTDR/data/imgs/best.png')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    parser.add_argument('--input_mae')
    parser.add_argument('--input_maes')
    parser.add_argument('--input_csv')
    parser.add_argument('--input_csvs')
    parser.add_argument('--info_csv')
    parser.add_argument('--output')
    parser.add_argument('--threshold',type=float)

    args = parser.parse_args()
    if args.task=='parse':
        mae_to_csv(args.input_mae,args.output)
    if args.task=='merge':
        merge_csvs(args.input_csvs,args.output,args.threshold)
    if args.task=='mae_split':
        mae_split(args.input_mae,args.input_csv,args.output)
    if args.task=='merge_properties':
        merge_properties(args.input_csv,args.info_csv,args.output)
    if args.task=='get_smiles':
        get_smiles(args.input_csv,args.info_csv,args.output)
    if args.task=='split_csv':
        split_csv(args.input_csv,args.output)
    if args.task=='sep_mae':
        sep_mae(args.input_mae,args.input_csvs,args.output)
    if args.task=='get_scores':
        get_scores(args.info_csv,args.input_maes,args.output)