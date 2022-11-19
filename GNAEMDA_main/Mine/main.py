import numpy as np
from inits import load_data
import argparse
import torch
import random
from Model import VGNAE

def main():
    # ------------------data_parameters-----------------#
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")], default=('net1'),help="lists of graphs to use.")
    parser.add_argument('--attributes', type=lambda s: [item for item in s.split(",")], default=('features,similarity'), help=" attributes ： features similarity")
    parser.add_argument('--data_path', type=str, default='MDAD', help="lists of dataset : MDAD  , DrugVirus, aBiofilm")
    parser.add_argument('--isolated_degree', type=list, default=[], help="degree of add isolated node in train data " )
    parser.add_argument('--k_folds', type=int, default='5', help="Cross verify the added isolated nodes")
    args_data = parser.parse_args()
    k = args_data.k_folds
    print(args_data)
    graph = 'net1'
    adj, Features, A, labels, num_drug, num_microbe = load_data(graph, args_data)
    Features = torch.tensor(Features.toarray(), dtype=torch.float32)

    # -----------------model_parameters-----------------#
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Epoch_Num', type=int, default='200', help="")
    parser.add_argument('--scale', type=int, default='0', help=" ")
    parser.add_argument('--Learning_Rate', type=float, default='5e-3', help="")
    parser.add_argument('--Hidden_Layer_1', type=int, default='3092', help="")
    parser.add_argument('--Hidden_Layer_2', type=int, default='256', help="")
    parser.add_argument('--scaling_factor', type=float, default=1.6)#
    args_model = parser.parse_args()

    # ---------------------k-fold_valid_experiment-----------------#
    from tools import cross_valid_experiment, get_class_edge, caculate_avg
    pos_test, neg_test, pos_test_Bip, neg_test_Bip = cross_valid_experiment(adj, k, labels, num_drug, num_microbe)

    Results_GNAEMDA = []
    Results_VGNAEMDA = []
    isolated_num = []

    for i in range(k):
        print('----------------this is',i+1,'th corss---------------')

        train_adj = np.array(adj, copy=True)
        test_edge = pos_test[i]
        test_false = neg_test[i]
        for index in test_edge:
            train_adj[index[0]][index[1]] = 0
            train_adj[index[1]][index[0]] = 0
        two_class_data = get_class_edge(train_adj, test_edge, test_false)#get isolated nodes and connected nodes
        isolated_num.append(two_class_data.isolated_node_num)

        print('-----GNAEMDA-----')
        result_GNAE = VGNAE(args_model, args_model.scaling_factor, adj, Features, labels, 'GAE',  train_adj, test_edge, test_false, two_class_data, num_drug, num_microbe)
        Results_GNAEMDA.append(result_GNAE)

        # print('-----VGNAEMDA-----')
        # result_VGNAE = VGNAE(args_model, args_model.scaling_factor, adj, Features, labels, 'VGAE', train_adj, test_edge, test_false, two_class_data, num_drug, num_microbe)
        # Results_VGNAEMDA.append(result_VGNAE)


    print('GNAEMDA_avg:', caculate_avg(Results_GNAEMDA))

    # print('VGNAEMDA_avg:', caculate_avg(Results_VGNAEMDA))
    # print(isolated_num)
    # print('mean:', np.mean(np.array(isolated_num)))
    # print('std:', np.std(np.array(isolated_num)))
    # print('percent: {:.2%}'.format(np.mean(np.array(isolated_num))/(num_drug+num_microbe)))

main()
