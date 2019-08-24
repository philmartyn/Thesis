# Borrowed heavily from https://github.com/marciahon29/Ryerson_MRP
# and https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
import io

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import normalize
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras import applications
from sklearn.preprocessing import LabelBinarizer
import pickle
from keras.regularizers import l2

# dimensions of our images.
img_width, img_height = 150, 150

# Paths to model output and data
output_base_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/Fold5/'
top_model_weights_path = output_base_path + 'bottleneck_weights.h5'
model_path = output_base_path + 'bipolar.model'
data_base_path = '/Users/pmartyn/PycharmProjects/Thesis/TransferLearning/Data/Crossfolds/Fold5/'
train_data_dir = data_base_path + 'converted/train'
validation_data_dir = data_base_path + 'converted/validation'
# Hyperparameters inputs relating to data
nb_train_samples = 4512
nb_validation_samples = 1280
init_lr = 1e-3
epochs = 50
batch_size = 16
data_array = []
labels_array = []


# Create and save feature arrays
# Bottleneck features in VGG16 model refers to the last activation maps before the fully connected layer.
def gen_and_save_bottleneck_features():
    imagegen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print("Model created. ", model)

    generator = imagegen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print("Generator created. ", generator)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print("bottleneck_features_train created. ", bottleneck_features_train)

    np.save(output_base_path + 'bottleneck_features_train.npy',
            bottleneck_features_train)

    generator = imagegen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(output_base_path + 'bottleneck_features_validation.npy',
            bottleneck_features_validation)


# Train the VGG16 fully connected layer using the data.
def train_top_model():
    train_data = np.load(open(output_base_path + 'bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    print("Training data. ", train_data.size)
    print("Training labels. ", train_labels.size)
    print("Training labels. ", train_labels)

    lb = LabelBinarizer()
    training_labels = lb.fit_transform(train_labels)

    print("Training labels post labeliser. ", training_labels.size)

    validation_data = np.load(open(output_base_path + 'bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    val_labels = lb.fit_transform(validation_labels)

    print("Validation shape ", validation_data.shape[1])

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, training_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, val_labels))
    model.save_weights(top_model_weights_path)
    model.save(model_path)

    with open(output_base_path + 'history.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


gen_and_save_bottleneck_features()
train_top_model()