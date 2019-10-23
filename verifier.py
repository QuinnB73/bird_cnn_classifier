from tensorflow.keras.models import load_model
import numpy as np
from cv2 import cv2
from gcloud_trainer.trainer.utils import load_data, load_categories
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('-i', '--input-image', help='The image to predict')

def load_trained_model():
    model = load_model('/Users/quinnbudan/school/fourth_year/trained_model.h5')
    print('Successfully loaded model')
    return model

def load_all_data():
    images, labels = load_data('/Users/quinnbudan/neural_nets/honours_project_work/data/training_data.npz')
    categories = load_categories('/Users/quinnbudan/neural_nets/honours_project_work/data/cat.txt')
    print('Successfully loaded images and categories')
    return images, labels, categories

def make_predictions(model, image, categories):
    img_array = cv2.imread(image, cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (256, 256))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img_array)
    cv2.waitKey(0)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict_classes(img_array, batch_size=1)
    bird = categories[prediction[0]]
    print(bird)

if __name__ == "__main__":
    args = argparser.parse_args()
    image_path = args.input_image

    model = load_trained_model()
    #images, labels, categories = load_all_data()
    categories = load_categories('/Users/quinnbudan/neural_nets/honours_project_work/data/cat.txt')
    make_predictions(model, image_path, categories)
