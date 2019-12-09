from tensorflow.keras.models import load_model
import numpy as np
from cv2 import cv2
from gcloud_trainer.trainer.utils import load_data, load_categories
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('-i', '--input-image', help='The image to predict', required=True)
argparser.add_argument('-c', '--categories', help='The categories file', required=True)
argparser.add_argument('-m', '--model', help='The model file', required=True)
argparser.add_argument('-s', '--size', help='The input image size', required=True)

def load_trained_model(model_path):
    #model = load_model('/Users/quinnbudan/school/fourth_year/trained_model.h5')
    model = load_model(model_path)
    print('Successfully loaded model')
    return model

def make_prediction(model, image, categories, size):
    """ This function loads the image, shows it on screen, makes a prediction
    using the model, and prints that prediction."""

    img_array = cv2.imread(image, cv2.IMREAD_COLOR)
    img_array = cv2.resize(img_array, (size, size))
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

    model = load_trained_model(args.model)
    #categories = load_categories('/Users/quinnbudan/neural_nets/honours_project_work/data/cat.txt')
    categories = load_categories(args.categories)
    make_prediction(model, args.input_image, categories, int(args.size))
