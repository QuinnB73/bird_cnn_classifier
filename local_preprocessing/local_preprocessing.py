import random
import os
import numpy as np
from ..gcloud_trainer.trainer import utils
from cv2 import cv2
from argparse import ArgumentParser

IMG_SIZE = 300
NUMPY_ARCHIVE = 'training_data_300.npz'

argparser = ArgumentParser()
argparser.add_argument('-d', '--data-dir', help='Load, preprocess, and save data')
argparser.add_argument('-r', '--num-random', help='Specify the number of random images from the dataset to show')
argparser.add_argument('-c', '--categories', help='The categories file')

def preprocess_data(data_dir, categories, num_to_show=0):
    """This function preprocesses the images of birds, scaling
    them all to a specific constant size (200x200). The labels and images are
    shuffled. Then, these preprocessed, shuffled images are saved into a 
    numpy array with the shape -1 x IMG_SIZE x IMG_SIZE x 3. Where -1 means an 
    length-inferred 1st dimension, and IMG_SIZE x IMG_SIZE x 3 is the "shape" 
    of the image. Note that the 3 here means that the images are in full RGB colour,
    necessary for differentiating between birds.

    If num_to_show is not 0, then that many times a random image will
    be shown from the scaled dataset.

    The shuffled training images and their labels are returned."""

    training_data = []

    for category in categories:
        class_num = categories.index(category)
        category = category.replace(' ', '-')
        path = os.path.join(data_dir, category)
        print(f'Beginning to preprocess data in {path}')

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception:
                pass

    if num_to_show != 0:
        while num_to_show > 0:
            num = random.randint(0, len(training_data))
            img_array, _ = training_data[num]
            cv2.imshow('image', img_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            num_to_show -= 1

    random.shuffle(training_data)
    images = []
    labels = []

    for image, label in training_data:
        images.append(image)
        labels.append(label)

    # Convert images to numpy array, shaped as our image size, with last argument being the number of colour channels.
    # Note: -1 means inferred 1st dimension, based on length of array
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    images = images / 255.0
    print('Data preprocessing complete')
    return images, labels

def save_training_data_numpy(images, labels):
    """This function saves images and training labels as a compressed
    .npz file."""
    np.savez_compressed(NUMPY_ARCHIVE, images=images, labels=labels)

    print(f'Done saving training data to {NUMPY_ARCHIVE}')

def main():
    args = argparser.parse_args()

    data_dir = args.data_dir
    categories_file = args.categories

    if not data_dir:
        print('Error, data directory to load images from must be specified')
    if not categories_file:
        print('Error, categories file must be specified')

    categories = utils.load_categories(categories_file)

    num_to_show = int(args.num_random) if args.num_random else 0
    images, labels = preprocess_data(data_dir, categories, num_to_show)
    save_training_data_numpy(images, labels)

if __name__ == "__main__":
    main()
