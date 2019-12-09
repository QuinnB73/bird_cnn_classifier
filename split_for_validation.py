import os
import shutil
import math
from argparse import ArgumentParser
from random import shuffle

argparser = ArgumentParser()
argparser.add_argument('-s', '--src-dir', help='The source directory to split into validation images')
argparser.add_argument('-d', '--dest-dir', help='The destination directory to put the new train test split into')
argparser.add_argument('-t', '--target-split', help='The target test/train split, e.g. 0.2', type=float)

def do_split_into_train_and_test(src_dir, train_path, test_path, target_split):
    """ This function randomly splits the dataset into train and test sets. """

    src_files = os.listdir(src_dir)
    train_dirs = []
    test_dirs = []
    total_train_images = 0
    total_test_images = 0

    for file_name in src_files:
        full_path = os.path.join(src_dir, file_name)
        if '.DS_Store' not in file_name:
            train_dir = make_new_dir(train_path, file_name)
            test_dir = make_new_dir(test_path, file_name)

            train_dirs.append(train_dir)
            test_dirs.append(test_dir)

            images = os.listdir(full_path)
            shuffle(images)
            num_images = len(images)
            num_test_images = math.floor(num_images * target_split)
            num_train_images = num_images - num_test_images

            total_train_images += num_train_images
            total_test_images += num_test_images

            # Copy the files into the test/train folder
            count = 0
            for image in images:
                full_image_path = os.path.join(full_path, image)
                dest_dir = train_dir if count <= num_train_images else test_dir
                shutil.copy2(full_image_path, dest_dir)
                count += 1

    print(f'Training images: {total_train_images} Testing images: {total_test_images}')
    print(f'Total images: {total_train_images + total_test_images}')


def make_new_dir(starting_path, new_dir_name):
    new_dir_path = os.path.join(starting_path, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    print(f'Made new directory {new_dir_path}')
    return new_dir_path
    

def main():
    args = argparser.parse_args()
    src_dir = args.src_dir
    dest_dir = args.dest_dir
    target_split = args.target_split

    train_path = make_new_dir(dest_dir, 'data/train')
    test_path = make_new_dir(dest_dir, 'data/test')
    do_split_into_train_and_test(src_dir, train_path, test_path, target_split)


if __name__ == "__main__":
    main()
