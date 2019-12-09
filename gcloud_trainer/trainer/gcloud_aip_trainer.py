import numpy as np
import sys, os, random, shutil, subprocess
import tensorflow as tf
from datetime import datetime
from . import utils
from . import basic_cnn
from google.cloud import storage
from argparse import ArgumentParser
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

def augmented_train_model(model, data_path, categories, training_config):
    """
    This function uses ImageDataGenerator to augment the data and fit the model.
    It then saves the model and uploads it to a GCloud bucket
    """
    is_split = False
    try:
        is_split = training_config['is_split']
    except KeyError:
        pass
    bucket_path = training_config['bucket_path']
    model_name = training_config['model_base_name']
    batch_size = training_config['batch_size']
    num_epochs = training_config['epochs']
    num_per_epoch = training_config['num_per_epoch']
    image_size = training_config['input_shape'][0]
    validation_percent = training_config['validation_percent']
    lr_plateau_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    model_name += '_{}'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    if not is_split:
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
            subset='validation')

        model.fit_generator(training_gen, steps_per_epoch=num_per_epoch, epochs=num_epochs,
            callbacks=[lr_plateau_cb], validation_data=validation_gen, validation_steps=num_per_valid,
            validation_freq=1)

    else:
        training_datagen = ImageDataGenerator(horizontal_flip=True, height_shift_range=0.2,
            width_shift_range=0.2, rotation_range=20, rescale=1./255)
        testing_datagen = ImageDataGenerator(rescale=1./255)

        train_path = '{}/train/'.format(data_path)
        test_path = '{}/test/'.format(data_path)

        num_per_valid = (num_per_epoch * validation_percent) // batch_size
        num_per_epoch = num_per_epoch // batch_size

        training_gen = training_datagen.flow_from_directory(directory=train_path, target_size=(image_size, image_size),
            color_mode='rgb', classes=categories, class_mode='sparse', shuffle=True, batch_size=batch_size)
        validation_gen = testing_datagen.flow_from_directory(directory=test_path, target_size=(image_size, image_size),
            color_mode='rgb', classes=categories, class_mode='sparse', shuffle=True, batch_size=batch_size)

        model.fit_generator(training_gen, steps_per_epoch=num_per_epoch, epochs=num_epochs,
            callbacks=[lr_plateau_cb], validation_data=validation_gen,  validation_steps=num_per_valid,
            validation_freq=1)

        num_per_valid = num_per_epoch * validation_percent
        val_loss, val_acc = model.evaluate_generator(validation_gen, steps=num_per_valid)
        
    # Save the model and upload it to GCloud bucket
    model.save(model_name)
    utils.save_file_to_bucket(bucket_path, model_name)
    return val_loss, val_acc

def k_fold_cv(k, data_path, categories, network_config, training_config, augmented):
    """
    This function performs k-fold cross validation, and prints the results of
    the k experiments.
    """
    training_dirs = ['train', 'test']
    losses = []
    accs = []
    training_config['is_split'] = True
    bucket_path = training_config['bucket_path']
    log_file_name = 'k_fold_cross_validation_log_{}.txt'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    if not augmented:
        print('The k-fold cross validation function only supports augmented training')
        return

    # Make train and test directories
    working_dir = os.getcwd()
    for training_dir in training_dirs:
        os.mkdir(os.path.join(working_dir, training_dir))
        for category in categories:
            os.mkdir(os.path.join(working_dir, training_dir, category))

    # Load and shuffle all image paths
    print('Starting to split data into {} pieces for k-fold cross validation'.format(k))
    img_paths = []
    for category in categories:
        cat_dir = os.path.join(data_path, category)
        for img in os.listdir(cat_dir):
            img_paths.append(os.path.join(cat_dir, img))

    random.shuffle(img_paths)

    # Split into 5 partitions
    img_paths = np.array(img_paths)
    partitions = np.array_split(img_paths, k)

    for i in range(k):
        print('Starting training experiment {}'.format(i + 1))

        # Build a new model
        model = build_model(network_config, training_config, len(categories))

        # Clear the two set directories
        for training_dir in training_dirs:
            for cat_dir in os.listdir(os.path.join(working_dir, training_dir)):
                for img in os.listdir(os.path.join(working_dir, training_dir, cat_dir)):
                    os.remove(os.path.join(working_dir, training_dir, cat_dir, img))

        # Copy partitions 0, ..., i - 1, i + 1, ..., k into train dir
        # and copy partition i into the test dir
        for j in range(k):
            for img_path in partitions[j]:
                # It's the directory before the image name
                category = img_path.split(os.sep)[-2]
                target_dir = training_dirs[0] if j != i else training_dirs[1]
                copy_target = os.path.join(working_dir, target_dir, category)
                shutil.copy2(img_path, copy_target)

        # Train with this split, storing the results
        val_loss, val_acc = augmented_train_model(model, working_dir, categories, training_config)
        losses.append(val_loss)
        accs.append(val_acc)

    with open(log_file_name, 'w') as log_file:
        log_file.write('{}\n'.format(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')))
        log_file.write('K-Fold cross validation with k = {}\n'.format(k))
        for i in range(k):
            val_loss = losses[i]
            val_acc = accs[i]
            msg = 'Experiment {} - loss: {} acc: {}'.format(i + 1, val_loss, val_acc)
            print(msg)
            log_file.write('{}\n'.format(msg))
    
    utils.save_file_to_bucket(bucket_path, log_file_name)

    # Delete created directories
    for training_dir in training_dirs:
        shutil.rmtree(training_dir)

def build_model(network_config, training_config, num_categories):
    # Network config
    num_convolutional_layers = network_config['num_convolutional_layers']
    num_filters = network_config['num_filters']
    kernel_size = tuple(network_config['kernel_size'])
    pool_size = tuple(network_config['pool_size'])
    dense_layer_size = network_config['dense_layer_size']
    dropout_rate = network_config['dropout_rate']
    batch_normalization = network_config['batch_normalization']
    small_first = network_config['small_first']

    input_shape = tuple(training_config['input_shape'])
    learning_rate = training_config['learning_rate']

    model = basic_cnn.build_model(input_shape, num_categories, num_convolutional_layers, num_filters,
        kernel_size, pool_size, dense_layer_size, dropout_rate, learning_rate, batch_normalization, small_first)
    return model


def main():
    args = argparser.parse_args()

    # Load the YAML configuration
    network_config, training_config = utils.load_yaml_config(args.config_file)

    # Training config
    augmented = training_config['augmented']
    data_path = training_config['data_path']
    categories_path = training_config['categories_path']

    categories = utils.load_categories(categories_path)

    # K-fold config
    k = -1
    try:
        k = training_config['k_fold']
    except KeyError:
        pass

    # Train, save, and upload the model
    if k > 0:
        data_dir = utils.unzip_data(data_path)
        k_fold_cv(k, data_dir, categories, network_config, training_config, augmented)
    else:
        # Build the model
        model = build_model(network_config, training_config, len(categories))
        if augmented:
            data_dir = utils.unzip_data(data_path)
            augmented_train_model(model, data_dir, categories, training_config)
        else:
            images, labels = utils.load_data(data_path)
            train_model(model, images, labels, training_config)

if __name__ == "__main__":
    main()
