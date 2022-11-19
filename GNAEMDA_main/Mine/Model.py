from pylab import *
from scipy.sparse import csr_matrix
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAE, APPNP, VGAE
import sys
import torch
from torch_geometric.data import Data

# warnings.filterwarnings('ignore')
sys.path.append('./Data_Process')
path_result = "./Latent_representation/"

def process_data(edge_index):
    A = np.array(edge_index)
    teams00 = list(A[0])
    teams11 = list(A[1])
    # print(teams0,teams1)
    matrix = np.zeros((1546, 1546))
    for i in range(len(teams00)):
        x = int(teams00[i])
        y = int(teams11[i])
        matrix[x][y] = 1
    return matrix

def process_adjTrain(data,num_drug, num_microbe):
    num_sum = num_drug + num_microbe
    team = csr_matrix((data), shape=(num_sum, num_sum))
    team = np.array(team[:(num_sum * num_sum), :num_sum].toarray() > 0, dtype=np.int)
    teams1 = team.nonzero()[1]
    teams0 = team.nonzero()[0]
    team = [teams0, teams1]
    data_tensor = torch.tensor(team)
    return data_tensor

def process_toTensor(data):
    processed_data = torch.Tensor(np.array(data).conj().T)
    return processed_data


def VGNAE(args_model,scaling_factor, Adjacency_Matrix_raw, Features, labels, choesn_model, train_adj, pos_test, neg_test,two_class_data,num_drug, num_microbe):
    # ---模型参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default= choesn_model)
    parser.add_argument('--dataset', type=str, default='MADA')
    parser.add_argument('--epochs', type=int, default = args_model.Epoch_Num)
    parser.add_argument('--scaling_factor', type=float, default= scaling_factor)
    args = parser.parse_args()
    train_adj = np.array(train_adj, copy=True)
    epochs = args_model.Epoch_Num
    features = Features
    data_tensor = process_adjTrain(train_adj, num_drug, num_microbe)
    test_edges = process_toTensor(pos_test)
    test_edges_false = process_toTensor(neg_test)
    test_connected_edge = process_toTensor(two_class_data.test_connected_edge)
    test_connected_false = process_toTensor(two_class_data.test_connected_false)
    test_isolated_edge = process_toTensor(two_class_data.test_isolated_edge)
    test_isolated_false = process_toTensor(two_class_data.test_isolated_false)

    data = Data(edge_index=data_tensor, x= features, test_mask=1373, train_mask=1373, val_mask=1373, y=1373)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        if args.model in ['VGAE']:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return loss, z

    def test(pos_edge_index, neg_edge_index, plot_his=0):
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels, edge_index):
            super(Encoder, self).__init__()
            self.linear1 = nn.Linear(in_channels, out_channels)
            self.linear2 = nn.Linear(in_channels, out_channels)
            self.propagate = APPNP(K=1, alpha=0)

        def forward(self, x, edge_index, not_prop=0):
            if args.model == 'GAE':
                x = self.linear1(x)
                x = F.normalize(x, p=2, dim=1) * args.scaling_factor
                x = self.propagate(x, edge_index)
                return x

            if args.model == 'VGAE':
                x_ = self.linear1(x)
                x_ = self.propagate(x_, edge_index)
                x = self.linear2(x)
                r_x = torch.Tensor(x)
                x = F.normalize(x, p=2, dim=1) * args.scaling_factor
                x = self.propagate(x, edge_index)

                return x + r_x, x_

            norm = torch.norm(x, p=2, dim=1)
            self.asd = norm
            return x

    channels = args_model.Hidden_Layer_2


    data_tensor1 = np.array(data_tensor)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == 'GAE':
       model = GAE(Encoder(data.x.size()[1], channels, data_tensor)).to(dev)
    if args.model == 'VGAE':
       model = VGAE(Encoder(data.x.size()[1], channels, data_tensor)).to(dev)

    x, train_pos_edge_index = data.x.to(dev), data_tensor.to(dev)


    optimizer = torch.optim.Adam(model.parameters(), lr=args_model.Learning_Rate)  # 使用adam来做梯度下降

    max_auc = 0
    max_ap = 0
    for epoch in range(0, epochs):

        loss, emb = train()
        loss = float(loss)

        with torch.no_grad():
               print('------------------------------------------')
               auc, ap, fpr, tpr = test(test_edges, test_edges_false)
               # roc_connected, ap_connected, fpr2, tpr2 = test(test_connected_edge, test_connected_false)
               # roc_isolated, ap_isolated, fpr3, tpr3 = test(test_isolated_edge, test_isolated_false)
               print('Epoch: {:03d}, LOSS: {:.5f}'.format(epoch, loss))
               print('test:','AUC: {:.5f}, AP: {:.5f}'.format(auc, ap))
               # print('connected_auc:{:.5f}, connected_ap:{:.5f}'.format(roc_connected, ap_connected))
               # print('isolated_auc:{:.5f}, isolated_ap:{:.5f}'.format(roc_isolated, ap_isolated))
               if auc > max_auc:
                   max_auc = auc
                   max_ap = ap

    print('max_auc:', max_auc, 'ap:', max_ap)
    return max_auc

