import tensorflow as tf
print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from utils import *
import os
import argparse
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=5,
        help='batch size')
parser.add_argument('--load_path', type=str,
        default='./saved_model/graphnn-model_best',
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')

def get_avr_topk_hitrate(distances, all_labels, tested_function_idx, my_label, topk=1):
    new_distances = np.delete(distances, tested_function_idx)
    all_labels = np.delete(all_labels, tested_function_idx)
    groups = [all_labels[np.where(new_distances == _)[0]] for _ in sorted(np.unique(new_distances))[:topk]]
    
    if not my_label in all_labels:
        return [-1] * topk

    k = []
    hit = 0
    for group in groups:
        if my_label in group:
            hit = 1
            while len(k) != topk:
                k.append(hit)
            break
        else:
            k.append(hit)
    return k

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    DATA_FILE_NAME = './data/acfgSSL_{}/'.format(NODE_FEATURE_DIM)
    SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
    OPTIMIZATION=('-O0', '-O1','-O2','-O3')
    COMPILER=('armeb-linux', 'i586-linux', 'mips-linux')
    VERSION=('v54',)

    FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
            OPTIMIZATION, VERSION)
    FUNC_NAME_DICT = get_f_dict(F_NAME)
     

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    print "{} graphs, {} functions".format(len(Gs), len(classes))


    if os.path.isfile('data/class_perm.npy'):
        perm = np.load('data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.8,0.0,0.2],perm)

    print "Train: {} graphs, {} functions".format(
            len(Gs_train), len(classes_train))
    print "Dev: {} graphs, {} functions".format(
            len(Gs_dev), len(classes_dev))
    print "Test: {} graphs, {} functions".format(
            len(Gs_test), len(classes_test))

    # Fix the pairs for validation and testing
    # if os.path.isfile('data/valid.json'):
    #     with open('data/valid.json') as inf:
    #         valid_ids = json.load(inf)
    #     valid_epoch = generate_epoch_pair(
    #             Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    # else:
    #     valid_epoch, valid_ids = generate_epoch_pair(
    #             Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
    #     with open('data/valid.json', 'w') as outf:
    #         json.dump(valid_ids, outf)

    # if os.path.isfile('data/test.json'):
    #     with open('data/test.json') as inf:
    #         test_ids = json.load(inf)
    #     test_epoch = generate_epoch_pair(
    #             Gs_test, classes_test, BATCH_SIZE, load_id=test_ids)
    # else:
    #     test_epoch, test_ids = generate_epoch_pair(
    #             Gs_test, classes_test, BATCH_SIZE, output_id=True)
    #     with open('data/test.json', 'w') as outf:
    #         json.dump(test_ids, outf)

    # valid_epoch, valid_ids = generate_epoch_pair(
    #         Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
    # with open('data/valid.json', 'w') as outf:
    #     json.dump(valid_ids, outf)

    # test_epoch, test_ids = generate_epoch_pair(
    #         Gs_test, classes_test, BATCH_SIZE, output_id=True)
    # with open('data/test.json', 'w') as outf:
    #     json.dump(test_ids, outf)

    # Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    # Test
    # val_auc, fpr, tpr, thres = get_auc_epoch(
    #         gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    # gnn.say( "AUC on validation set: {}".format(val_auc) )
    # test_auc, fpr, tpr, thres = get_auc_epoch(
    #         gnn, Gs_test, classes_test, BATCH_SIZE, load_data=test_epoch)
    # gnn.say( "AUC on testing set: {}".format(test_auc) )

    
    embeds = []
    all_labels = []
    feature_dim = len(Gs[0].features[0])

    for graph in tqdm(Gs):
        all_labels.append(graph.label)
        X = np.zeros((1, graph.node_num, feature_dim))          # expects batch size for getting embedding so batch is set to 1
        mask = np.zeros((1, graph.node_num, graph.node_num))

        for node in range(graph.node_num):
            X[0, node, :] = np.array(graph.features[node])
            for edge in graph.succs[node]:
                mask[0, node, edge] = 1

        embeds.append(gnn.get_embed(X, mask).squeeze())               # get_embed is their own function, requires X and mask
                
    all_labels = np.asarray(all_labels)
    embeds = np.asarray(embeds)

    topk_precision = 10
    test_hits = [0] * topk_precision
    test_able_hits = len(Gs_test)
    
    for test_graph in tqdm(Gs_test):
        k = []
        curr_function_idx = Gs.index(test_graph)
        label = test_graph.label

        distances = euclidean_distances([embeds[curr_function_idx]], embeds) # 1 by len(Gs_test)
        distances = distances.squeeze()   

        topk_hits = get_avr_topk_hitrate(distances, all_labels, curr_function_idx, label, topk=topk_precision)

        if all(hit == -1 for hit in topk_hits):
            test_able_hits -= 1

        else:
            for idx, hit in enumerate(topk_hits):
                test_hits[idx] += hit

    print("Test set hitrates: ")
    for k in range(1, topk_precision+1):
        print("p@%d: %f" % (k, test_hits[k-1]/float(test_able_hits)))
