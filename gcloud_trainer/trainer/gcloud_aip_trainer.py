## TODO: Add k-fold training, validation, and then testing regime

import numpy as np
import sys
import tensorflow as tf
from datetime import datetime
from . import utils
from . import basic_cnn
from google.cloud import storage
from argparse import ArgumentParser
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ReduceLROnPlateau

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

IMG_SIZE = 200

argparser = ArgumentParser()
argparser.add_argument('-c', '--config-file', help='The name of the bucket to save the model in')
argparser.add_argument('-j', '--job-dir', help='GCloud bucket path to save checkpoints') # TODO: Implement checkpoints

def train_model(model, images, labels, training_config):
    """This function fits the model and returns it with the weights and biases updated."""
    batch_size = training_config['batch_size']
    num_epochs = training_config['epochs']
    validation_percent = training_config['validation_percent']

    lr_plateau_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

    # scikit-learn shuffle the two NP arrays together
    images, labels = shuffle(images, labels)
    model.fit(images, labels, batch_size=batch_size, validation_split=validation_percent,
        epochs=num_epochs, callbacks=[lr_plateau_cb])
    return model

def main():
    args = argparser.parse_args()

    # Load the YAML configuration
    network_config, training_config = utils.load_yaml_config(args.config_file)

    # Network config
    num_convolutional_layers = network_config['num_convolutional_layers']
    num_filters = network_config['num_filters']
    kernel_size = tuple(network_config['kernel_size'])
    pool_size = tuple(network_config['pool_size'])
    dense_layer_size = network_config['dense_layer_size']
    dropout_rate = network_config['dropout_rate']

    # Training config
    data_path = training_config['data_path']
    categories_path = training_config['categories_path']
    bucket_path = training_config['bucket_path']
    model_name = training_config['model_base_name']
    learning_rate = training_config['learning_rate']
    dropout_rate_str = str(dropout_rate).split('.')[1]
    dropout_rate_str += '0' if len(dropout_rate_str) == 1 else ''
    model_name += '_l{}_f{}_k{}_p{}_d{}_r{}_{}'.format(num_convolutional_layers, num_filters,
        kernel_size[0], pool_size[0], dense_layer_size, dropout_rate_str, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    # Build the model
    images, labels = utils.load_data(data_path)
    categories = utils.load_categories(categories_path)
    model = basic_cnn.build_model(images, categories, num_convolutional_layers, num_filters,
        kernel_size, pool_size, dense_layer_size, dropout_rate, learning_rate)

    # Train the model
    model = train_model(model, images, labels, training_config)

    # Save the model and upload it to GCloud bucket
    model.save(model_name)
    utils.save_model_to_bucket(bucket_path, model_name)

if __name__ == "__main__":
    main()
