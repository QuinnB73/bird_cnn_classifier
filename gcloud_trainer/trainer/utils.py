import numpy as np
import re
import sys
import yaml
import tarfile
from google.cloud import storage

NUMPY_ARCHIVE = 'training_data.npz'

def load_yaml_config(config_file):
    """This function loads a .yml or .yaml configuration file to configure
    the neural network and the training.
    
    The configuration file is expected to have a network_configuration dictionary and
    a training_configuration dictionary.
    
    Returns a tuple of two dictionaries, one for the network configuration and
    one for the training configuration."""
    if 'gs://' in config_file:
        download_file(config_file, 'config.yml')
        config_file = 'config.yml'

    with open(config_file, 'r') as config:
        try:
            configuration = yaml.safe_load(config)
            network_config = configuration['network_config']
            training_config = configuration['training_config']

            return network_config, training_config
        except Exception as error:
            print('Error loading config: {}'.format(error))


def load_categories(categories_file):
    """This function loads a .txt file that contains one category per line,
    these are read into a list and returned.
    
    Will throw an exception if the file does not exist."""
    if 'gs://' in categories_file:
        download_file(categories_file, 'categories.txt')
        categories_file = 'categories.txt'

    x = []
    with open(categories_file, 'r') as cat_f:
        x = [line.strip('\n') for line in cat_f]
    
    return x

def load_data(data_file):
    """This function loads a .npz file to use for predictions.
    If the data_file argument passed in is a file in a GCloud bucket,
    this function will download it to training_data.npz and load it.
    
    Will throw an exception if the file does not exist."""
    if 'gs://' in data_file:
        download_file(data_file, NUMPY_ARCHIVE)
        data_file = NUMPY_ARCHIVE

    data = np.load(data_file)
    images = data['images']
    labels = data['labels']

    print('Successfully loaded data from {}'.format(data_file))
    return images, labels

def unzip_data(data_file):
    """This function unzips a .tar.gz and extracts it to a directory,
    returning that directory. If the file is in a GCloud bucket, this
    function will download it first."""
    if 'gs://' in data_file:
        data_file = download_file(data_file, None)
    
    with tarfile.open(data_file, 'r:gz') as data_tar:
        data_tar.extractall()

    unzipped_name = re.search('[^\.]*', data_file).group(0)
    return unzipped_name

def download_file(remote_fname, local_fname):
    """This function downloads the file from the given GCloud bucket URL and returns
    the name of the file that the content was downloaded to.

    The input URL should be of the format gs://<bucket_name>/path/to/file."""
    print('Received GCloud bucket path {}'.format(remote_fname))

    bucket_name = re.search('//(.*?)/', remote_fname).group(1)
    path_to_file = re.search('//.*?/(.*)', remote_fname).group(1)
    if not local_fname:
        local_fname = re.search('[^/]+\.tar\.gz', remote_fname).group(0)
    
    print('Extracted bucket name {} and path {} from URL {}'.format(bucket_name, path_to_file, remote_fname))

    print('Attempting to download data from GCloud bucket {}'.format(remote_fname))
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(path_to_file)
        blob.download_to_filename(local_fname)
        
        print('Successfully downloaded data from GCloud bucket {} to {}'.format(remote_fname, local_fname))
        return local_fname
    except Exception as e:
        print('Error downloading data from GCloud bucket {}: {}'.format(remote_fname, e))
        sys.exit(2)

def save_file_to_bucket(bucket_path, file_name):
    """This function uploads the file to the provided
    GCloud bucket under the same name."""
    print('Will attempt to upload {} to {}'.format(file_name,  bucket_path))

    bucket_name = re.search('//(.*?)/', bucket_path).group(1)
    path_to_file = re.search('//.*?/(.*)', bucket_path).group(1)

    path_to_file += '/{}'.format(file_name)

    retries = 0
    while retries < 3:
        try:
            retries += 1
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(path_to_file, chunk_size=26214400)
            blob.upload_from_filename(file_name)
        except Exception as e:
            print('Error uploading {} to {}: {}'.format(file_name, bucket_path, e))
            sys.exit(2)

    print('Successfully uploaded {} to {}'.format(path_to_file, bucket_path))