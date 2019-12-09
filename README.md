# bird-cnn-classifier

This repository contains all of the Python modules used to build and train
the neural network on the Google Cloud AI Platform in the gcloud_trainer
submodule. In order to run any of the Python scripts in the gcloud_trainer
submodule locally, it is necessary to run the script as a Python 2 module
with this directory as the top-level module so that dependencies can function
properly. Note that all of the scripts in the gcloud_trainer submodule are
Python 2 compatible, wheras most of the other scripts are not. This is
because those scripts run on the Google Cloud AI Platform, which does not
support Python 3.

It also contains the scripts used to build the dataset in the retriever
submodule. Note that this submodule requires having the Flickr API key
environment variables set to work properly. Supporting scripts such as a
script to compute statistics for K-Fold Cross Validation results or to
provide a sanity check on the validity of the network are included at the top
level.

The example_configs directory contains example YAML configuration files to
use to train the network. 
