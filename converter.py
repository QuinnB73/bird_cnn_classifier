import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator

argparser = ArgumentParser()
argparser.add_argument('-m', '--model', help='The model to convert')
argparser.add_argument('-o', '--output-file', help='The file to save the tflite model to')

def do_convert(model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    tflite_model = converter.convert()

    return tflite_model

def save_lite_model(tflite_model, output_filename):
    with open(output_filename, 'wb') as output_file:
        output_file.write(tflite_model)

def main():
    args = argparser.parse_args()
    model_path = args.model
    output_file = args.output_file

    tflite_model = do_convert(model_path)
    save_lite_model(tflite_model, output_file)

if __name__ == "__main__":
    main()