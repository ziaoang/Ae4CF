from collections import defaultdict
import tensorflow as tf
import numpy as np
import random
import math
import sys
from load import load

#try:
#    batch_size = int(sys.argv[1])
#    learn_rate = float(sys.argv[2])
#except:
#    print("batch_size learn_rate")
#    exit()

batch_size = 64
learn_rate = 0.05

# hyper parameter
k = 500
epoch_count = 100

# load data
dataset = "ml-1m"
train_set, test_set = load(dataset)
max_user_id = max([t[0] for t in train_set])
input_size = max_user_id + 1

# change format
train_data = defaultdict(lambda:[0.0]*input_size)
train_mask = defaultdict(lambda:[0.0]*input_size)
for t in train_set:
    user_id, item_id, rating = t
    train_data[item_id][user_id] = rating
    train_mask[item_id][user_id] = 1.0

test_data = defaultdict(lambda:[0.0]*input_size)
test_mask = defaultdict(lambda:[0.0]*input_size)
for t in test_set:
    user_id, item_id, rating = t
    test_data[item_id][user_id] = rating
    test_mask[item_id][user_id] = 1.0

# predict data
pre_data, pre_data_, pre_mask_ = [], [], []
for item_id in test_data:
    pre_data.append(train_data[item_id])
    pre_data_.append(test_data[item_id])
    pre_mask_.append(test_mask[item_id])

# auto encoder
data = tf.placeholder(tf.float32, [None, input_size])
mask = tf.placeholder(tf.float32, [None, input_size])

scale = math.sqrt(6.0 / (max_user_id + 1 + k))

w1 = tf.Variable(tf.random_uniform([input_size, k], -scale, scale))
b1 = tf.Variable(tf.random_uniform([k], -scale, scale))
mid = tf.nn.softmax(tf.matmul(data, w1) + b1)

w2 = tf.Variable(tf.random_uniform([k, input_size], -scale, scale))
b2 = tf.Variable(tf.random_uniform([input_size], -scale, scale))
y = tf.matmul(mid, w2) + b2

data_ = tf.placeholder(tf.float32, [None, input_size])
mask_ = tf.placeholder(tf.float32, [None, input_size])
rmse = tf.sqrt(tf.reduce_sum(tf.square((y - data_)*mask_)) / tf.reduce_sum(mask_))
mae = tf.reduce_sum(tf.abs((y - data_)*mask_)) / tf.reduce_sum(mask_)

# training
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

loss = tf.reduce_mean(tf.reduce_sum(tf.square((y - data)*mask), 1))
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

# iterate
random.seed(123456789)
item_id_list = train_data.keys()
for epoch in range(epoch_count):
    random.shuffle(item_id_list)
    
    # train
    for batch_id in range( len(item_id_list) / batch_size ):
        start = batch_id * batch_size
        end = start + batch_size

        batch_data = []
        batch_mask = []
        for i in range(start, end):
            item_id = item_id_list[i]
            batch_data.append(train_data[item_id])
            batch_mask.append(train_mask[item_id])

        train_step.run(feed_dict={data:batch_data, mask:batch_mask})

    # predict
    rmse_score = rmse.eval(feed_dict={data:pre_data, data_:pre_data_, mask_:pre_mask_})
    mae_score = mae.eval(feed_dict={data:pre_data, data_:pre_data_, mask_:pre_mask_})
    print("%.4f\t%.4f"%(rmse_score, mae_score))



