import tensorflow as tf 
import numpy as np 
import cv2 
import json
from memoir.batch_preprocessing.channels import change_channel
import random
import memoir.data.batch_generator as bg
import os
from memoir.autoencoders import dense_network_directory

file_path = os.path.dirname(os.path.realpath(__file__)) 
os.chdir(file_path)

class Autoencoder:

    '''
    Dense Autoencoder class. Builds a tensorflow graph according to user entered model- and hyper-parameters.

    Args:
        name(str): Name of the object. This name will be used to name the directory of outputs.
        learning_rate(float): Learning rate of the model. Default: 1e-3
        loss_function(str): Loss function to be used for training. Default: 'mse'
        batch_size(int): Batch size of the model input. Default: 64
        epochs(int): Number of epochs to execute the graph for. Default: 500
        optimizer(str): Optimizer to be used for training. Default: 'adam'
        image_size(tuple): Size of the input image, irrespective of channels. Default: (100, 100)
        dropout_keep_prob(float): Rate of dropout layer. Default: 0.5
        channel_of_input(str): Channels of the input to used. Default: 'gray'

    Returns:
        An object of the class `Autoencoder` with the user entered attributes.
    '''

    def __init__(
        self,
        name="Youd_Should_Name_Things",
        learning_rate=1e-3,
        loss_function='mae',
        batch_size=64,
        epochs=500,
        optimizer='adam',
        image_size=(100, 100),
        dropout_keep_prob=0.5,
        channel_of_input='gray',
        **kwargs
        ):

        self.name = name
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.image_size = image_size
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.channel_of_input = channel_of_input

        for key, value in kwargs.items():
            setattr(self, key, value)

        with open('./paths.json', 'r') as file:
            self.paths = json.load(file)

        with open(self.paths["path_to_architecture"], 'r') as file:
            self.architecture = json.load(file)

        self.number_of_layers = len(list(self.architecture))
        self.available_activations = dense_network_directory.avail_activations(_return_=True)
        self.available_initializers = dense_network_directory.avail_initializers(_return_=True)
        self.available_optimizers = dense_network_directory.avail_optimizers(_return_=True)
        self.available_regularizers = dense_network_directory.avail_regularizers(_return_=True)
        self.available_loss_functions = dense_network_directory.avail_metrics(_return_=True)

    def build_graph(self, _return_=False, _print_=False):

        '''
        This method builds the graph according to the architecture written in `dense_network.json` file.

        Args:
            _return_(bool): Whether to return the constructed graph ot not. Helpful when you want to use the graph for some other task. Default: False
            _print_(bool): Whether to print the constructed graph for inspection. Default: False

        Returns:
            Nothing unless `_return` is set to True.
        '''
        
        self.layer_outputs = {}

        input_shape = self.image_size[0] * self.image_size[1]
        output_shape = input_shape * 3

        self.X = tf.placeholder(dtype=tf.float32, shape=(None, input_shape), name='Gray_Image')
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None, output_shape), name='RGB_Image')
        self.is_training = tf.placeholder_with_default(False, (), name='Flag_for_Training')
        self.keep_prob = tf.placeholder_with_default(1.0, (), name='Keep_Prob')

        previous_layer = self.X

        for i in range(1, self.number_of_layers + 1):

            layer_dict = self.architecture['Layer_' + str(i)]

            if i == self.number_of_layers:
                layer_dict['units'] = output_shape

            for layer in self.architecture['Layer_' + str(i)]['order']:
                layer_name = 'Layer_' + str(i) + '_' + layer

                if layer == 'dense':

                    self.layer_outputs[layer_name] = tf.layers.dense(
                                                                    inputs=previous_layer,
                                                                    units=layer_dict['units'],
                                                                    kernel_initializer=self.available_initializers[layer_dict['kernel_init']],
                                                                    bias_initializer=self.available_initializers[layer_dict['bias_init']],
                                                                    kernel_regularizer=self.available_regularizers[layer_dict['kernel_regularizer']],
                                                                    bias_regularizer=self.available_regularizers[layer_dict['bias_regularizer']],
                                                                    name=layer_name
                                                                    )
        
                if layer == 'activation':

                    self.layer_outputs[layer_name] = self.available_activations[layer_dict['activation_fn']](previous_layer, name=layer_name)


                if layer == 'dropout':

                    self.layer_outputs[layer_name] = tf.layers.dropout(
                                                                    inputs=previous_layer,
                                                                    rate=1-tf.cast(self.keep_prob, tf.float32),
                                                                    training=self.is_training,
                                                                    name=layer_name
                                                                    )

                if layer == 'batch_norm':

                    self.layer_outputs[layer_name] = tf.layers.batch_normalization(
                                                                                inputs=previous_layer,
                                                                                momentum=0.9,
                                                                                training=self.is_training,
                                                                                name=layer_name
                                                                                )
                previous_layer = self.layer_outputs[layer_name]
                
        if _return_:
            return self.layer_outputs
        if _print_:
            print('\n')
            for key in self.layer_outputs.keys():
                print(self.layer_outputs[key])

    def fit(self, publish_summaries=True, save_checkpoints=True, interval_of_checkpoints=5, save_outputs=True, output_interval=5, max_keep_checkpoints=10):

        '''
        Execute the built graph. For now it is only capable of fitting over our dataset.

        Args:
            publish_summaries(bool): Whether to write summaries of training to be visualized in Tensorboard. Set the path for summaries in `paths.json` file. Default: True
            save_checkpoints(bool): Whether to save model training checkpoints. Helpful when you want to restart the training or while testing the model. Default: True
            interval_of_checkpoints(int): Interval in terms of epochs to save the model checkpoints at. Default: 5
            save_outputs(bool): Whether to save the model outputs or not. Default: True
            output_interval(int): Interval in terms of epochs to save model outputs at. Default: 5
            max_keep_checkpoints(int): Number of checkpoints to store. As new checkpoints come, the old ones will be deleted. Default: 10

        Returns:
            Nothing
        '''

        ########################################## GPU_GROWTH #########################################

        tf_config=tf.ConfigProto()
        tf_config.gpu_options.allow_growth=True

        ###################################### LOSS FUNCTION(S) #######################################

        main_train_loss = tf.reduce_mean(self.available_loss_functions[self.loss_function](
                                                                            y_true=self.Y,
                                                                            y_pred=self.layer_outputs[list(self.layer_outputs)[-1]]
                                                                            ))

        ########################################## SUMMARIES ##########################################
        
        if publish_summaries:
            summaries = []
            summaries.append(tf.summary.scalar('Loss', main_train_loss))
            for layer_num in range(1, self.number_of_layers + 1):
                # kernels, biases = {}, {}
                # kernels['Layer_' + str(layer_num)] = tf.get_default_graph().get_tensor_by_name('Layer_' + str(layer_num) + '_dense/kernel:0')
                # biases['Layer_' + str(layer_num)] = tf.get_default_graph().get_tensor_by_name('Layer_' + str(layer_num) + '_dense/bias:0')
                summaries.append(tf.summary.histogram('Layer_' + str(layer_num) + 'dense_kernel', tf.get_default_graph().get_tensor_by_name('Layer_' + str(layer_num) + '_dense/kernel:0')))
                summaries.append(tf.summary.histogram('Layer_' + str(layer_num) + 'dense_bias', tf.get_default_graph().get_tensor_by_name('Layer_' + str(layer_num) + '_dense/bias:0')))
            merged_summary = tf.summary.merge(summaries)

        ########################################## OPTIMIZER ##########################################

        if self.optimizer == 'adagradDA':
            global_step = tf.compat.v1.train.get_global_step()
            train_step = self.available_optimizers['adagradDA'](learning_rate=self.learning_rate, global_step=global_step).minimize(loss=main_train_loss)
        else:
            train_step = self.available_optimizers[self.optimizer](learning_rate=self.learning_rate).minimize(loss=main_train_loss)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.Session(config=tf_config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            image_names = bg.image_names_generator()
            batches = len(image_names) // self.batch_size

            writer = tf.summary.FileWriter(os.path.join(self.paths['path_to_tensorflow_data'], self.name, 'Tensorflow/LOGS/'))
            writer.add_graph(sess.graph)

            if save_checkpoints:
                saver = tf.train.Saver(max_to_keep=max_keep_checkpoints)

            for epoch in range(1, self.epochs + 1):
                for batch_num in range(1, batches + 1):

                    train_batch = bg.batch_generator(image_names, batch_size=self.batch_size, image_size=self.image_size)
                    
                    train_batch_gray = change_channel(batch=train_batch, channel=self.channel_of_input)

                    train_batch_reshaped = np.reshape(train_batch, (self.batch_size, -1))
                    train_batch_gray_reshaped = np.reshape(train_batch_gray, (self.batch_size, -1))

                    feed_dict_train = {
                        self.X: train_batch_gray_reshaped,
                        self.Y: train_batch_reshaped,
                        self.is_training: True,
                        self.keep_prob: self.dropout_keep_prob
                    }
                    _, _, loss = sess.run([train_step, extra_update_ops, main_train_loss], feed_dict=feed_dict_train)
                    
                    if publish_summaries:
                        s = sess.run(merged_summary, feed_dict=feed_dict_train)
                        writer.add_summary(s, epoch)
                        writer.flush()

                    print('Epoch: {}/{} - Batch: {}/{}        Loss: {}'.format(epoch, self.epochs, batch_num, batches, loss))
                
                    if save_outputs:
                    
                        if epoch % output_interval == 0 and batch_num == 1:
                            out_ = sess.run(self.layer_outputs[list(self.layer_outputs)[-1]], feed_dict=feed_dict_train)
                            out_reshaped = np.reshape(out_, (self.batch_size, self.image_size[0], self.image_size[1], 3))
                            out_reshaped = np.clip(out_reshaped * 255, 0, 255)
                            out_reshaped = out_reshaped.astype(np.uint8)

                            random_index = random.randint(0, self.batch_size - 1)
                            input_sample = cv2.cvtColor(train_batch_gray[random_index], cv2.COLOR_GRAY2BGR)

                            final_image = np.concatenate([input_sample, train_batch[random_index], out_reshaped[random_index]], axis=1)
                            
                            output_path = os.path.join(self.paths["model_output_paths"], self.name, 'OUTPUTS/')
                            
                            try:
                                if not os.path.exists(output_path):
                                    os.mkdir(output_path)
                            except OSError:
                                raise OSError('Could not make directory to save model outputs.')

                            cv2.imwrite(output_path + 'Output_epoch_' + str(epoch) + '_.jpg', final_image)
                    
                    if save_checkpoints:
                        if epoch % interval_of_checkpoints == 0 and batch_num == 1:
                            checkpoint_path = os.path.join(self.paths['path_to_tensorflow_data'], self.name, 'Tensorflow/Checkpoints/')
                            saver.save(sess, checkpoint_path + 'Epoch_{}'.format(epoch))
                        

            writer.close()