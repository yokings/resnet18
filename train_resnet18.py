import os

import numpy as np
import tensorflow as tf
import resnet18
from data_generator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import tensorflow.contrib.slim as slim
import facenet
import time



BATCH_SIZE = 4
EPOCH_SIZE = 380
MAX_NROF_EPOCHS = 10

IMAGE_H = 112
IMAGE_W = 96

CENTER_LOSS_FACTOR = 0.003
CENTER_LOSS_ALFA = 0.6

KEEP_PROBABILITY = 0.8
EMBEDDING_SIZE = 128
WEIGHT_DECAY = 0.00005

LEARNING_RATE = 0.1
LEARNING_RATE_DECAY_EPOCHS = 16
LEARNING_RATE_DECAY_FACTOR = 0.1

MOVING_AVERAGE_DECAY = 0.9999

PRETRAINED_MODEL = './pretrain'

GPU_MEMORY_FRACTION = 0.95
num_classes=10
display_step=100
train_set_dir='image_label.txt'
logdir='./log'
checkpoint_path='./modelfiles'
trainlog=open('train_log.txt','w')

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_set_dir,
                                 batch_size=BATCH_SIZE,
                                 num_classes=num_classes,
                                 shuffle=True)
    print(tr_data)
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
    training_init_op = iterator.make_initializer(tr_data.data)


    global_step = tf.Variable(0, trainable=False)

    
    x = tf.placeholder(tf.float32, [BATCH_SIZE, 112, 96, 3])
    y = tf.placeholder(tf.float32, [BATCH_SIZE, num_classes])
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate') 


    network=resnet18
    prelogits= network.inference(x, embedding_size=EMBEDDING_SIZE, phase_train=True, weight_decay=WEIGHT_DECAY)
    logits = slim.fully_connected(prelogits, num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
                                      scope='Logits', reuse=False)
    print(logits.shape)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    learning_rate_op = tf.train.exponential_decay(learning_rate_placeholder, global_step, LEARNING_RATE_DECAY_EPOCHS*EPOCH_SIZE, LEARNING_RATE_DECAY_FACTOR, staircase=True)
    with tf.name_scope('softmax_loss'):
    	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy_per_example')
    	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    with tf.name_scope('total_loss'):
    	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    	total_loss = tf.add_n([cross_entropy_mean] + regularization_losses)
    with tf.name_scope('train'):
    	opt = tf.train.RMSPropOptimizer(LEARNING_RATE)
    	grads = opt.compute_gradients(total_loss, tf.global_variables())
    	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    	variables_averages_op = variable_averages.apply(tf.trainable_variables())
    	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    		train_op = tf.no_op(name='train')
    	merged_summary = tf.summary.merge_all()
    	saver = tf.train.Saver(tf.trainable_variables())




gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logdir, sess.graph)
    # Add the model graph 
    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      logdir))
    for epoch in range(MAX_NROF_EPOCHS):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        trainlog.write("{} Epoch number: {}".format(datetime.now(), epoch+1)+'\n')
        lr = LEARNING_RATE

        sess.run(training_init_op)

        for step in range(EPOCH_SIZE):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)
            # And run the training op
            start_time = time.time()
            err, _, step, lrt=sess.run([total_loss,train_op,global_step,learning_rate_op], feed_dict={x: img_batch,y: label_batch,learning_rate_placeholder: lr})
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tlr: %f\tTime %.3f\tLoss %2.3f' % (epoch, step+1, EPOCH_SIZE, lrt, duration, err))
        os.makedirs(checkpoint_path + '/epoch' + str(int(epoch+1)))
        checkpoint_dir = os.path.join(model_dir + '/epoch' + str(int(epoch+1)), 'model-%d.ckpt' % (epoch+1))
        print('saving model into checkpoint_path:' + checkpoint_dir)
        saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=batch_number)

            
