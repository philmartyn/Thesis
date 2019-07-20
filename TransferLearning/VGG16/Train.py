# Borrowed heavily from https://github.com/marciahon29/Ryerson_MRP

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.preprocessing import LabelBinarizer
import pickle


# dimensions of our images.
img_width, img_height = 240, 240

# Paths to model output and data
top_model_weights_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/bottleneck_weights.h5'
model_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/bipolar.model'
train_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/converted/train'
validation_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/converted/validation'
# Hyperparameters inputs relating to data
nb_train_samples = 6112
nb_validation_samples = 1440
init_lr = 1e-3
epochs = 50
batch_size = 40
data_array = []
labels_array = []


# Create and save feature arrays
# Bottleneck features in VGG16 model refers to the last activation maps before the fully connected layer.
def gen_and_save_bottleneck_features():
    imagegen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(240,240,3))
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

    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = imagegen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


# Train the VGG16 fully connected layer using the data.
def train_top_model():

    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    print("Training data. ", train_data.size)
    print("Training labels. ", train_labels.size)

    lb = LabelBinarizer()
    training_labels = lb.fit_transform(train_labels)

    print("Training labels post labeliser. ", training_labels.size)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    val_labels = lb.transform(validation_labels)

    # Output the pickle file for use later in prediction
    f = open("/Users/pmartyn/PycharmProjects/Thesis/Output/bipolar_lb.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, val_labels))
    model.save_weights(top_model_weights_path)
    model.save(model_path)


# gen_and_save_bottleneck_features()
train_top_model()
