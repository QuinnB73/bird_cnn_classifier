from . import utils
from argparse import ArgumentParser
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

argparser = ArgumentParser()
argparser.add_argument('-d', '--data-file', help='The path to the data file to load')
argparser.add_argument('-c', '--categories-file', help='The path to the categories file to load')
argparser.add_argument('-m', '--model-name', help='The name of the model to save')
argparser.add_argument('-l', '--conv-layers', help='The number of convolutional layers',
    type=int, default=6)
argparser.add_argument('-f', '--filters', help='The number of filters in the convolutional layers',
    type=int, default=64)
argparser.add_argument('-k', '--kernel-size', help='The size of the filters in the convolutional layers',
    nargs=2, default=[3, 3], type=int)
argparser.add_argument('-p', '--pool-size', help='The size of the pooling window',
    nargs=2, default=[2, 2], type=int)
argparser.add_argument('-s', '--dense-layer-size', help='The number of neurons in the final fully connected layer',
    type=int, default=64)
argparser.add_argument('-r', '--dropout-rate', help='Add a dropout rate', type=float, default=0.0)
argparser.add_argument('-lr', '--learning-rate', help='The learning rate', type=float, default=0.001)

def build_model(data, categories, n_conv_layers, n_filters, k_size, p_size, d_size, do_rate, lrate):
    """This function constructs a simple CNN. The input layer is implicit
    based on the input data.

    Inputs:
        data - The training data (what is important here is the shape of the tensor)
        categories - The categories which defines the number of output neurons
        n_conv_layers - The number of convolutional layers
        n_filters - The number of convolutional filters per convolutional layer
        k_size - The size of the convolutional filters, must be tuple (x, y)
        p_size - The size of the max pooling window, must be tuple (x, y)
        d_size - The number of neurons in the final fully connected layer
        do_rate - The dropout rate in the fully connected layer
    
    Each convolutional layer has the same number and size of filters, and each
    MaxPooling "layer" has the same pool size."""

    msg = 'This is a network with {} convolutional layers, each with {} filters '.format(n_conv_layers, n_filters)
    msg += 'of size {} and each followed by MaxPooling with a pool size of {} '.format(k_size, p_size)
    msg += 'followed by a fully connected dense layer with a dropout rate of {} with {} neurons.'.format(do_rate, d_size)
    msg += ' This network will be trained with learning_rate={}'.format(lrate)
    print(msg)
    model = Sequential()

    # Build convolutional layers
    for i in range(0, n_conv_layers):
        if i == 0:
            model.add(Conv2D(32, k_size, padding='same', input_shape=data.shape[1:]))
        else:
            model.add(Conv2D(n_filters, k_size, padding='same'))

        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=p_size))

    # Dense layer followed by output layer
    model.add(Flatten())
    model.add(Dense(d_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Add dropout
    if do_rate:
        model.add(Dropout(do_rate))

    # Output Layer
    model.add(Dense(len(categories)))
    model.add(Activation('softmax'))

    # Compile
    adam_optimizer = Adam(lr=lrate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    return model

def main():
    args = argparser.parse_args()
    images, _ = utils.load_data(args.data_file)
    categories = utils.load_categories(args.categories_file)
    n_conv_layers = args.conv_layers
    n_filters = args.filters
    k_size = tuple(args.kernel_size)
    p_size = tuple(args.pool_size)
    d_size = args.dense_layer_size
    do_rate = args.dropout_rate
    lrate = args.learning_rate
    
    model = build_model(images, categories, n_conv_layers, n_filters, k_size, p_size, d_size, do_rate, lrate)
    print(model.summary())

if __name__ == "__main__":
    main()