import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from dgllife.utils import EarlyStopping, Meter
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
import dgl
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

# 设置全局随机种子
import os
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats = atom_featurizer.feat_size('hv')
e_feats = bond_featurizer.feat_size('he')

from dgllife.data import MoleculeCSVDataset

# last_csv1='ini_test_score.csv'
# last_csv2='iter1_score.csv'
total_csv='iter2_total.csv'
df=pd.read_csv(total_csv)
# df2=pd.read_csv(last_csv1)
# df3=pd.read_csv(last_csv2)
smiles_set=set()
ind_list=[]
for i, smiles in enumerate(df['SMILES']):
    if smiles not in smiles_set:
        smiles_set.add(smiles)
        ind_list.append(i)
print(len(ind_list))
import random
val_list=random.sample(ind_list,len(ind_list)//10)
train_list=[]
for i in ind_list:
    if i not in val_list:
        train_list.append(i)
dc_listings1=df.iloc[train_list]
# dc_listings1 = pd.concat([df2,df3,df.iloc[train_list]]) 
# dc_listings1=dc_listings1.iloc[:len(dc_listings1)//5]
dc_listings2 = df.iloc[val_list]
def load_data(data,name,load):
    dataset = MoleculeCSVDataset(data,
                                 smiles_to_graph=smiles_to_bigraph,
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer= bond_featurizer,
                                 smiles_column='SMILES',
                                 cache_file_path=str(name)+'_dataset.bin',
                                 task_names=['5NTP_XP','6QU7_XP'],
                                 load=load,init_mask=True,n_jobs=8
                            )
    return dataset
train_datasets = load_data(dc_listings1,f'utils/train_{total_csv}',True)
val_datasets = load_data(dc_listings2,f'utils/val_{total_csv}',True)
train_loader = DataLoader(train_datasets, batch_size=256,shuffle=True,
                          collate_fn=collate_molgraphs)
vali_loader = DataLoader(val_datasets,batch_size=256,shuffle=True,
                          collate_fn=collate_molgraphs)


def run_a_train_epoch(n_epochs, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        batch_data
        smiles, bg, labels, masks = batch_data
        bg=bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg, n_feats, e_feats)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())
    total_r2 = np.mean(train_meter.compute_metric('r2'))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training_r2 {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_r2,total_loss))
    return total_r2, total_loss

def run_an_eval_epoch(n_epochs, model, data_loader,loss_criterion):
    model.eval()
    val_losses=[]
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction = model(bg, n_feats, e_feats)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_loss=val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric('r2'))
        total_loss = np.mean(val_losses)
    return total_score, total_loss

model = model_zoo.AttentiveFPPredictor(node_feat_size=n_feats,
                                   edge_feat_size=e_feats,
                                   num_layers=2,
                                   num_timesteps=1,
                                   graph_feat_size=300,
                                   n_tasks=2,
                                   dropout=0.3
                                    )
model = model.to(device)

#Train
loss_fn = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.000001
                             )
stopper = EarlyStopping(mode='higher', patience=20)
n_epochs = 501
for e in range(n_epochs):
    score = run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
    val_score = run_an_eval_epoch(n_epochs, model, vali_loader,loss_fn)
    early_stop = stopper.step(val_score[0], model)
    if val_score[0]==stopper.best_score:
        torch.save(model.state_dict(), f'data/models/dgl/dgl.pt')
    if e % 10 == 0:
        print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, best validation {} {:.4f}'.format(
        e + 1, n_epochs, 'r2', val_score[0], 'loss', val_score[-1],
        'r2', stopper.best_score))
    if early_stop:
        break


# if __name__ == '__main__':
#     fn = 'MTATFP_jak.pt'
#     torch.save(model.state_dict(), fn)


