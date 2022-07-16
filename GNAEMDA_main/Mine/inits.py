import numpy as np
import scipy.io as sio
import scipy.sparse as sp

def caculate_G_degree(A):
    import networkx as nx
    G = nx.DiGraph(A)
    list_Degree = []
    for i in range(len(A)):
        list_Degree.append(int(G.degree(i)/2))
    return list_Degree

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)
    return feat_norm

def load_data(graph_type,args):

    print('loading adj...')
    P = {}
    P_v ={}
    data_path = args.data_path

    if data_path == 'MDAD':
       if graph_type == "net1":
           print('chose:net1')
           P = sio.loadmat('../data/'+data_path+'/net1.mat')
           P_v = np.array(P['interaction'])
       elif graph_type == "net2":
           print('chose:net2')
           P = sio.loadmat('../data/net2.mat')
           P_v = P['net2']
       labels = P_v

    if data_path == 'aBiofilm':
        print('Loading '+data_path+' dataset...')
        labels = np.loadtxt('../data/'+data_path+'/adj.txt')
        temp_label = np.zeros((1720, 140))
        for temp in labels:
            temp_label[int(temp[0]) - 1, int(temp[1]) - 1] = int(temp[2])
        labels = np.array(temp_label)
        P_v = labels

    if data_path == 'DrugVirus':
        print('Loading ' + data_path + ' dataset...')
        labels = np.loadtxt('../data/' + data_path + '/adj.txt')
        temp_label = np.zeros((175, 95))
        for temp in labels:
            temp_label[int(temp[0]) - 1, int(temp[1]) - 1] = int(temp[2])
        labels = np.array(temp_label)
        P_v = labels


    attributes_list = []
    A = np.array(labels)
    print('loading attributes...')
    for attribute in args.attributes:
        if attribute == 'features':
            F1 = np.loadtxt("../data/" + data_path + "/drug_features.txt")
            F2 = np.loadtxt("../data/" + data_path + "/microbe_features.txt")
            feature = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                                 np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))
            attributes_list.append(feature)
        elif attribute == 'similarity':
            F1 = np.loadtxt("../data/" + data_path + "/drug_similarity.txt")
            F2 = np.loadtxt("../data/" + data_path + "/microbe_similarity.txt")
            similarity = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                                    np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))
            attributes_list.append(similarity)
    features = np.hstack(attributes_list)
    features = normalize_features(features)
    features = sp.csr_matrix(features)

    num_drug = F1.shape[0]
    num_microbe = F2.shape[0]

    P = np.vstack((np.hstack((np.zeros(shape=(num_drug,num_drug),dtype=int), P_v)),np.hstack((P_v.transpose(),np.zeros(shape=(num_microbe, num_microbe),dtype=int)))))


    return P, features,A, labels, num_drug, num_microbe




