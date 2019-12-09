import os, re, sys
from argparse import ArgumentParser
from cv2 import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

argparser = ArgumentParser()
argparser.add_argument('-i', '--image', help='The image to augment', required=True)
argparser.add_argument('-o', '--output_dir', help='The directory in which to save the transformed images',
    required=True)

def augment_image(image_path, out_dir):
    """ This function augments the passed in image with three
    transformations. First it horizontally flips the image, then it rotates
    it by 20 degrees, and finally, it horizontally flips it, rotates it by 5
    degrees, and translates it on the y-axis by 50 units.
    
    The three resultant images are saved to hf_{image_name}, rt_{image_name},
    and tl_{image_name} in the provided directory, respectively."""

    # Setup ImageDataGenerator, the normal parameters don't matter since we will manually
    # apply transforms
    image_path = os.path.abspath(image_path)
    out_dir = os.path.abspath(out_dir)
    image_filename_pattern = r'[^/]+$'
    image_filename = re.search(image_filename_pattern, image_path)
    if not image_filename:
        print('Unable to extract filename from the provided path.')
        sys.exit(1)
    image_filename = image_filename.group(0)

    datagen = ImageDataGenerator(rescale=1./255)
    transform_params = {}

    # Load the image using OpenCV2
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    _, width, _ = img_array.shape

    # Apply horizontal flip
    transform_params['flip_horizontal'] = True
    horizontal_image = datagen.apply_transform(img_array, transform_params)
    transform_params.clear()

    # Apply 20 degree rotation
    transform_params['theta'] = 20.0
    rotated_image = datagen.apply_transform(img_array, transform_params)
    transform_params.clear()

    # Apply translation
    transform_params['flip_horizontal'] = True
    transform_params['theta'] = 15.0
    transform_params['ty'] = width * 0.2
    translated_image = datagen.apply_transform(img_array, transform_params)
    transform_params.clear()

    # Save images
    did_succeed = True
    did_succeed = did_succeed and cv2.imwrite(os.path.join(out_dir, image_filename), img_array)
    did_succeed = did_succeed and cv2.imwrite(os.path.join(out_dir, f'hf_{image_filename}'), horizontal_image)
    did_succeed = did_succeed and cv2.imwrite(os.path.join(out_dir, f'rt_{image_filename}'), rotated_image)
    did_succeed = did_succeed and cv2.imwrite(os.path.join(out_dir, f'tl_{image_filename}'), translated_image)

    if did_succeed:
        print(f'Successfully saved images to {out_dir}')
    else:
        print(f'Unable to save images to {out_dir}')
    
def main():
    args = argparser.parse_args()
    augment_image(args.image, args.output_dir)

if __name__ == "__main__":
    main()
