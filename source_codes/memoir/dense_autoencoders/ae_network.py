import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import memoir.data.batch_generator as bg
import memoir.data.data_augmentation as da
import channels
import json
import cv2

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
learning_rate=1e-4
epochs=500
batch_size=64
vid_type='Animated'
series='All'
image_size = (70, 70)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
image_names = bg.image_names_generator(v_type=vid_type, series=series)

batches = len(image_names) // batch_size

sample_batch = bg.batch_generator(image_names, batch_size=1, image_size=image_size)
sample_gray_batch = channels.bgr2gray(sample_batch)

sample_gray_batch = np.reshape(sample_gray_batch, (1, -1)) 
sample_batch = np.reshape(sample_batch, (1, -1))
input_shape = sample_gray_batch.shape[1]
output_shape = input_shape * 3

avail_inits = available_initializers(_return_=True)
avail_actis = available_activations(_return_=True)
avail_regul = available_regularizers(_return_=True)
avail_optis = available_optimizers(_return_=True)
avail_metri = available_metrics(_return_=True)

print(input_shape)

with tf.variable_scope('Placeholders'):

    X = tf.placeholder(tf.float32, shape=(None, input_shape), name='Input')
    Y = tf.placeholder(tf.float32, shape=(None, output_shape), name='Output')
    is_training = tf.placeholder_with_default(False, (), name='Batch_Flag')
    keep_prob = tf.placeholder_with_default(1.0, (), name='Dropout_Flag')

with tf.variable_scope('Encoder'):

    kernel_initializer = tf.keras.initializers.glorot_normal()
    bias_initializer = tf.keras.initializers.random_normal()
    kernel_regularizer = None
    bias_regularizer = None

    encoder_1 = tf.layers.dense(X, 1500, name='encoder_1', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    encoder_1_bn = tf.layers.batch_normalization(encoder_1, training=is_training, momentum=0.9, name='encoder_1_bn')
    encoder_1_ac = tf.nn.relu(encoder_1_bn, name='encoder_1_ac')
    encoder_1_drop = tf.nn.dropout(encoder_1_ac, keep_prob=keep_prob, name='encoder_1_drop')

    encoder_2 = tf.layers.dense(encoder_1_ac, 1000, name='encoder_2', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    encoder_2_bn = tf.layers.batch_normalization(encoder_2, training=is_training, momentum=0.9)
    encoder_2_ac = tf.nn.relu(encoder_2_bn)
    encoder_2_drop = tf.nn.dropout(encoder_2_ac, keep_prob=keep_prob, name='encoder_2_drop')

    encoder_3 = tf.layers.dense(encoder_2_ac, 500, name='encoder_3', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    encoder_3_bn = tf.layers.batch_normalization(encoder_3, training=is_training, momentum=0.9)
    encoder_3_ac = tf.nn.relu(encoder_3_bn)
    encoder_3_drop = tf.nn.dropout(encoder_3_ac, keep_prob=keep_prob, name='encoder_3_drop')

    encoder_4 = tf.layers.dense(encoder_3_ac, 300, name='encoder_4', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    encoder_4_bn = tf.layers.batch_normalization(encoder_4, training=is_training, momentum=0.9)
    encoder_4_ac = tf.nn.relu(encoder_4_bn)
    encoder_4_drop = tf.nn.dropout(encoder_4_ac, keep_prob=keep_prob, name='encoder_4_drop')


with tf.variable_scope('Decoder'):

    kernel_initializer = tf.keras.initializers.glorot_normal()
    bias_initializer = tf.keras.initializers.random_normal()
    kernel_regularizer = None
    bias_regularizer = None

    decoder_1 = tf.layers.dense(encoder_4_drop, 500, name='decoder_1', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    decoder_1_bn = tf.layers.batch_normalization(decoder_1, training=is_training, momentum=0.9)
    decoder_1_ac = tf.nn.relu(decoder_1_bn)
    decoder_1_drop = tf.nn.dropout(decoder_1_ac, keep_prob=keep_prob, name='decoder_1_drop')

    decoder_2 = tf.layers.dense(decoder_1_ac, 1000, name='decoder_2', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    decoder_2_bn = tf.layers.batch_normalization(decoder_2, training=is_training, momentum=0.9)
    decoder_2_ac = tf.nn.relu(decoder_2_bn)
    decoder_2_drop = tf.nn.dropout(decoder_2_ac, keep_prob=keep_prob, name='decoder_2_drop')

    decoder_3 = tf.layers.dense(decoder_2_ac, 1500, name='decoder_3', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    decoder_3_bn = tf.layers.batch_normalization(decoder_3, training=is_training, momentum=0.9)
    decoder_3_ac = tf.nn.relu(decoder_3_bn)
    decoder_3_drop = tf.nn.dropout(decoder_3_ac, keep_prob=keep_prob, name='decoder_3_drop')

    decoder_4 = tf.layers.dense(decoder_3_ac, output_shape, name='decoder_4', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
    decoder_4_ac = tf.nn.relu(decoder_4)


with tf.variable_scope('Loss'):
    loss = tf.reduce_mean(tf.squared_difference(decoder_4_ac, Y))

with tf.variable_scope('Train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.variable_scope('Layer_name'):
    weight_e1 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_1' + '/kernel:0')
    weight_e2 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_2' + '/kernel:0')
    weight_e3 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_3' + '/kernel:0')
    weight_e4 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_4' + '/kernel:0')
    
    weight_d1 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_1' + '/kernel:0')
    weight_d2 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_2' + '/kernel:0')
    weight_d3 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_3' + '/kernel:0')
    weight_d4 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_4' + '/kernel:0')

    bias_e1 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_1' + '/bias:0')
    bias_e2 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_2' + '/bias:0')
    bias_e3 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_3' + '/bias:0')
    bias_e4 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_4' + '/bias:0')
    
    bias_d1 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_1' + '/bias:0')
    bias_d2 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_2' + '/bias:0')
    bias_d3 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_3' + '/bias:0')
    bias_d4 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_4' + '/bias:0')

with tf.variable_scope('Summary'):
    tf.summary.scalar('Total_Loss', loss)
    tf.summary.histogram('E1', weight_e1)
    tf.summary.histogram('E2', weight_e2)
    tf.summary.histogram('E3', weight_e3)
    tf.summary.histogram('E4', weight_e4)

    tf.summary.histogram('D1', weight_d1)
    tf.summary.histogram('D2', weight_d2)
    tf.summary.histogram('D3', weight_d3)
    tf.summary.histogram('D4', weight_d4)
    
    tf.summary.histogram('E1', bias_e1)
    tf.summary.histogram('E2', bias_e2)
    tf.summary.histogram('E3', bias_e3)
    tf.summary.histogram('E4', bias_e4)

    tf.summary.histogram('D1', bias_d1)
    tf.summary.histogram('D2', bias_d2)
    tf.summary.histogram('D3', bias_d3)
    tf.summary.histogram('D4', bias_d4)

merged_summary = tf.summary.merge_all()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('/home/andy/Projects/Memoir/source_codes/memoir/autoencoders/Tensorflow/LOGS/')
    writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=10)

    for epoch in range(1, epochs+1):
        for batch in range(batches):
            
            train_batch = bg.batch_generator(image_names, batch_size=batch_size, image_size=image_size)
            train_batch_gray = channels.change_channel(train_batch, 'gray')
            train_batch, train_batch_gray = np.reshape(train_batch, (batch_size, -1)), np.reshape(train_batch_gray, (batch_size, -1))

            _, _, tr_l, s = sess.run([train_step, extra_update_ops, loss, merged_summary], feed_dict={X: train_batch_gray, Y: train_batch, is_training: True, keep_prob: 0.5})
            writer.add_summary(s, epoch)

            out_ = sess.run([decoder_4_ac], feed_dict={X: train_batch_gray, Y: train_batch, is_training: True, keep_prob: 0.5})
            out_ = np.reshape(out_, (batch_size, image_size[0], image_size[1], 3))

        print('Epoch: {0}/{1}      Loss: {2}'.format(epoch, epochs, tr_l))
        if epoch % 5 == 0:
            saver.save(sess, '/home/andy/Projects/Memoir/source_codes/memoir/autoencoders/Tensorflow/Checkpoints/epoch_{}'.format(epoch))