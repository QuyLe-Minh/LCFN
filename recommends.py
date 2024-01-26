import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from model_LCFN import model_LCFN

def top_k_recommends(sess, model, para_test):
    [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH] = para_test
    items_in_train_data = np.zeros((1, item_num))
    user_top_items = sess.run(model.top_items, feed_dict={model.users: [5], model.keep_prob: 1, model.items_in_train_data: items_in_train_data, model.top_k: 5})
    return user_top_items