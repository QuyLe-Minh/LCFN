import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
import random as rd
from model_LGCN import model_LGCN
import numpy as np
from params import all_para, pred_dim
from change_params import change_params
from read_data import read_all_data
import os

def train(user_batch, para, data):
    [train_data, train_data_interaction, user_num, item_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, _] = data
    [_, _, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN, _, TOP_K] = para[0:13]
    [_, _, _, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING] = para[13:]
    
    model = model_LGCN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())    
    
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))
    
    for batch_num in range(len(batches) - 1):
        train_batch_data = []
        for sample in range(batches[batch_num], batches[batch_num + 1]):
            (user, pos_item) = train_data_interaction[sample]
            sample_num = 0
            while sample_num < 1:
                neg_item = int(rd.uniform(0, item_num))
                if not (neg_item in train_data[user]):
                    sample_num += 1
                    train_batch_data.append([user, pos_item, neg_item])
        train_batch_data = np.array(train_batch_data)
        _, loss = sess.run([model.updates, model.loss], feed_dict={model.users: train_batch_data[:, 0], model.pos_items: train_batch_data[:, 1], model.neg_items: train_batch_data[:, 2], model.keep_prob: KEEP_PORB if MODEL == 'LGCN' else 1})    
    
    
    items_in_train_data = np.zeros((user_batch.shape[0], item_num))
    user_top_items = sess.run(model.top_items, feed_dict={model.users: user_batch, model.keep_prob: 1, model.items_in_train_data: items_in_train_data, model.top_k: 5})
    return user_top_items

def recommend(user_batch):
    change_dic = {
        'ACTIVATION': ['None', 'Tanh', 'Sigmoid', 'ReLU'][0],
        'dataset': 1,   # 0:Amazon, 1:Movielens
        'model': 8,     # 0:MF, 1:NCF, 2:GCMC, 3:NGCF, 4:SCF, 5:CGMC, 6:LightGCN, 7:LCFN, 8:LGCN, 9:SGNN
    }
    all_param = change_params(all_para, change_dic, pred_dim)
    
    para = all_param[0: 13]
    para += all_param[13: 26]
    all_param[11] = para[11] = 'Test'
    data = read_all_data(all_param)
    para[10] = data[-1]

    os.environ["CUDA_VISIBLE_DEVICES"] = all_param[0] 
    user_top_items = train(user_batch, para, data)
    return user_top_items

# if __name__ == '__main__':
#     print(recommend(np.array([4])))
    