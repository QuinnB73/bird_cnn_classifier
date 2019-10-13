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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import subprocess

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

def augmented_train_model(model, data_path, categories, image_size, training_config):
    """This function uses ImageDataGenerator to augment the data and fit the model."""
    batch_size = training_config['batch_size']
    num_epochs = training_config['epochs']
    num_per_epoch = training_config['num_per_epoch']
    validation_percent = training_config['validation_percent']
    lr_plateau_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

    num_per_valid = num_per_epoch * validation_percent
    num_per_epoch = (num_per_epoch - num_per_valid) // batch_size
    num_per_valid = num_per_valid // batch_size

    datagen = ImageDataGenerator(horizontal_flip=True, height_shift_range=0.2, width_shift_range=0.2,
        rotation_range=20, rescale=1./255, validation_split=validation_percent)

    training_gen = datagen.flow_from_directory(directory=data_path, target_size=(image_size, image_size),
        color_mode='rgb', classes=categories, class_mode='sparse', shuffle=True, batch_size=batch_size,
        subset='training')
    validation_gen = datagen.flow_from_directory(directory=data_path, target_size=(image_size, image_size),
        color_mode='rgb', classes=categories, class_mode='sparse', shuffle=True, batch_size=batch_size,
        subset='training')

    model.fit_generator(training_gen, steps_per_epoch=num_per_epoch, epochs=num_epochs,
        callbacks=[lr_plateau_cb], validation_data=validation_gen, validation_steps=num_per_valid,
        validation_freq=1)
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
    batch_normalization = network_config['batch_normalization']
    small_first = network_config['small_first']

    # Training config
    augmented = training_config['augmented']
    data_path = training_config['data_path']
    input_shape = tuple(training_config['input_shape'])
    categories_path = training_config['categories_path']
    bucket_path = training_config['bucket_path']
    model_name = training_config['model_base_name']
    learning_rate = training_config['learning_rate']
    dropout_rate_str = str(dropout_rate).split('.')[1]
    dropout_rate_str += '0' if len(dropout_rate_str) == 1 else ''
    model_name += '_l{}_f{}_k{}_p{}_d{}_r{}_{}'.format(num_convolutional_layers, num_filters,
        kernel_size[0], pool_size[0], dense_layer_size, dropout_rate_str, datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    # Build the model
    categories = utils.load_categories(categories_path)
    model = basic_cnn.build_model(input_shape, len(categories), num_convolutional_layers, num_filters,
        kernel_size, pool_size, dense_layer_size, dropout_rate, learning_rate, batch_normalization, small_first)

    # Train the model
    if augmented:
        data_dir = utils.unzip_data(data_path)
        model = augmented_train_model(model, data_dir, categories, input_shape[0], training_config)
    else:
        images, labels = utils.load_data(data_path)
        model = train_model(model, images, labels, training_config)

    # Save the model and upload it to GCloud bucket
    model.save(model_name)
    utils.save_model_to_bucket(bucket_path, model_name)

if __name__ == "__main__":
    main()
