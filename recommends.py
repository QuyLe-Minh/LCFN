import numpy as np
import tensorflow as tf
from model_LCFN import model_LCFN

def top_k_recommends(user_id, sess, model, para_test):
    [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH] = para_test
    para_test_one_user = [test_data, TOP_K] 
    items_in_train_data = np.zeros((1, item_num))
    TOP_K = 5
    user_top_items = sess.run(model.top_items, feed_dict={model.users: user_id, model.keep_prob: 1, model.items_in_train_data: items_in_train_data, model.top_k: TOP_K})
    return user_top_items