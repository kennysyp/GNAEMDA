import numpy as np
import math
from scipy.sparse import csr_matrix
import random

def caculate_avg(Results):
    sum = 0
    for i in Results:
        sum = sum + i
    return sum/len(Results)

def caulate_avg_ROC(all):
    avg_roc = []
    for i in range(all.shape[1]):
        clo = []
        for j in range(all.shape[0]):
            clo.append(all[j][i])
        num = caculate_avg(clo)
        avg_roc.append(num)
    avg_roc = np.array(avg_roc)
    print('avg_roc.shape:', avg_roc.shape)
    return avg_roc


def isolated_nodes_list(adj, isolated_nodes_degree, k_folds):
    print('-------------------------')
    D_list = caculate_G_degree(adj)
    import math
    connected_node_true, index_list = delet_edge_inA(D_list, adj, isolated_nodes_degree, 1373)
    length = len(index_list)
    lists = index_list
    index_list_all = []
    for i in range(k_folds):
        index_list = lists[math.floor(i / k_folds * length):math.floor((i + 1) / k_folds * length)]
        index_list_all.append(index_list)

    return index_list_all


from scipy.sparse import csr_matrix
from Data_Process import *

def rondom_split(k, arr_data):
    arr = range(len(arr_data))
    every_len = int(len(arr) / k)
    arr_flag = []
    random_num = []
    index = 0
    for i in range(len(arr)):
        arr_flag.append(True)
        random_num.append(index)
        index += 1

    random.shuffle(random_num)

    result_arr = []
    every_arr = []
    index = 0
    for i in range(0, len(arr) - 1, every_len):
        index += 1
        for j in range(every_len):
            every_arr.append(arr[random_num[i]])
            i += 1
        result_arr.append(every_arr)
        every_arr = []
        if index >= k:
            break

    for i in range(len(random_num) - len(result_arr) * every_len):
        result_arr[i].append(arr[random_num[len(arr) - 1 - i]])
    all = []
    for index in result_arr:
        list1 = []
        for i in index:
            list1.append(arr_data[i])
        all.append(list1)
    return all

def cross_valid_experiment(adj, k, labels,num_drug, num_microbe):
    pos_all = []
    neg_all = []
    pos_all_bip = []
    neg_all_bip = []
    for i in range(num_drug):
        for j in range(num_drug, num_drug + num_microbe):
            if adj[i][j] == 1:
                pos_all.append([i, j])
                pos_all_bip.append([i, j - num_drug])
            else:
                neg_all.append([i, j])
                if j > num_drug:
                    neg_all_bip.append([i, (j - num_drug)])
    length = len(pos_all)
    pos_list_all = rondom_split(k, pos_all)
    pos_test = []
    neg_test = []
    for item in pos_list_all:
        pos_test.append(item)
        neg_test.append(random.sample(list(neg_all), len(item)))

    length = len(pos_all_bip)
    pos_list_all_Bip = rondom_split(k, pos_all_bip)
    pos_test_Bip = []
    neg_test_Bip = []
    for item in pos_list_all_Bip:
        pos_test_Bip.append(item)
        neg_test_Bip.append(random.sample(list(neg_all_bip), len(item)))

    return pos_test, neg_test,pos_test_Bip,neg_test_Bip


def get_class_edge(train_adj, test_edges, test_edges_false):
    data = np.ones(train_adj.shape[0])
    adj_train = csr_matrix((data, (train_adj[:, 0], train_adj[:, 1])), shape=train_adj.shape)
    processed_A = train_adj
    list_degree = np.array(caculate_G_degree(processed_A))
    index_node = []
    for i in range(len(list_degree)):
        if list_degree[i] == 0:
            index_node.append(i)
    print('isolated_num:', len(index_node))
    test_connected_edge_all = []
    test_isolated_edge = []
    for item in test_edges:
        if item[0] in index_node or item[1] in index_node:
            test_isolated_edge.append(list(item))
        else:
            test_connected_edge_all.append(list(item))
    isolated_edges_num = len(list(test_isolated_edge))
    test_isolated_false = random.sample(list(test_edges_false), isolated_edges_num)
    for i in range(len(test_isolated_false)):
        test_isolated_false[i] = list(test_isolated_false[i])

    if len(test_connected_edge_all) > isolated_edges_num:
        test_connected_edge = random.sample(list(test_connected_edge_all), isolated_edges_num)
    else:
        test_connected_edge = test_connected_edge_all
    test_connected_false = random.sample(list(test_edges_false), isolated_edges_num)
    for i in range(len(test_isolated_false)):
        test_isolated_false[i] = list(test_isolated_false[i])

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--isolated_node_num', default=len(index_node), help='')
    parser.add_argument('--adj_train', default=adj_train, help="")
    parser.add_argument('--test_edges', default=test_edges, help="")
    parser.add_argument('--test_edges_false', default=test_edges_false, help="")
    parser.add_argument('--test_isolated_edge', default=test_isolated_edge, help="")
    parser.add_argument('--test_isolated_false', default=test_isolated_false, help="")
    parser.add_argument('--test_connected_edge', default=test_connected_edge, help="")
    parser.add_argument('--test_connected_false', default=test_connected_false, help="")
    args_data = parser.parse_args()
    return args_data

def change_to_Bip(list_edge,num_drug, num_microbe):
    for i in range(len(list_edge)):
        list_edge[i][0] = list_edge[i][0] - num_drug
        list_edge[i][1] = list_edge[i][1] -num_microbe
    return list_edge

def get_class_edge_bip(train_adj, test_edges, test_edges_false, num_drug, num_microbe):
    P_v = train_adj
    P = np.vstack((np.hstack((np.zeros(shape=(num_drug, num_drug), dtype=int), P_v)),np.hstack((P_v.transpose(), np.zeros(shape=(num_microbe, num_microbe), dtype=int)))))
    processed_A = P
    list_degree = np.array(caculate_G_degree(processed_A))
    index_node = []
    for i in range(len(list_degree)):
        if list_degree[i] == 0:
            index_node.append(i)
    print('isolated_num:', len(index_node))
    test_connected_edge_all = []
    test_isolated_edge = []
    for item in test_edges:
        if item[0] in index_node:
            test_isolated_edge.append(list(item))
        else:
            test_connected_edge_all.append(list(item))
    isolated_edges_num = len(list(test_isolated_edge))
    test_isolated_false = random.sample(list(test_edges_false), isolated_edges_num)
    for i in range(len(test_isolated_false)):
        test_isolated_false[i] = list(test_isolated_false[i])

    if len(test_connected_edge_all) > isolated_edges_num:
        test_connected_edge = random.sample(list(test_connected_edge_all), isolated_edges_num)
    else:
        test_connected_edge = test_connected_edge_all
    test_connected_false = random.sample(list(test_edges_false), isolated_edges_num)
    for i in range(len(test_isolated_false)):
        test_isolated_false[i] = list(test_isolated_false[i])

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adj_train', default=train_adj, help="")
    parser.add_argument('--test_edges', default=test_edges, help="")
    parser.add_argument('--test_edges_false', default=test_edges_false, help="")
    parser.add_argument('--test_isolated_edge', default=test_isolated_edge, help="")
    parser.add_argument('--test_isolated_false', default=test_isolated_false, help="")
    parser.add_argument('--test_connected_edge', default=test_connected_edge, help="")
    parser.add_argument('--test_connected_false', default=test_connected_false, help="")
    args_data = parser.parse_args()

    return args_data

def equal_list(list1,list2):
    if list1[0]+1 == list2[0] and list1[1]+1 == list2[1]:
        return True
    else:
        return False

def list_split(items, n):
    n = int(n)
    return [items[i:i + n] for i in range(0, len(items), n)]


