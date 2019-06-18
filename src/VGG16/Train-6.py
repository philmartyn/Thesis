import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras import applications

# dimensions of our images.
from keras.regularizers import l2
from keras.models import load_model
import cv2

img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_00.h5'
top_model_path = 'bottleneck_model.h5'
train_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/converted/train'
validation_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/converted/validation'
nb_train_samples = 4520
nb_validation_samples = 1280
epochs = 100
batch_size = 40



def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(1))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))

    # model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))
    # model.add(Dense(256, activation='relu', )) #kernel_initializer='he_normal'
    # model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    # model.add(Dropout(1))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation="relu", )) #kernel_initializer='he_normal'
    # model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)))


    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    model.save(top_model_path)

def predict():
    model = load_model('/Users/pmartyn/PycharmProjects/Thesis/src/VGG16/bottleneck_model.h5')
    other_model = applications.VGG16(include_top=False, weights='imagenet')

    image = cv2.imread('/Users/pmartyn/PycharmProjects/Thesis/Data/test/converted/train/bipolar/bipolar-slice-146.jpg')
    output = image.copy()
    image = cv2.resize(image, (img_width, img_width))

    # scale the pixel values to [0, 1]
    image = image.astype("float") / 255.0

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    features = other_model.predict(image)
    preds = model.predict_classes(features)

    i = preds.argmax(axis=1)[0]
    # print("[INFO] Label...", i)
    # print("[INFO] Label...", preds[0][i] * 100)
    # print("[INFO] Label...", preds)
    # print preds
    # print preds[0]
    print (preds[0][0])


# save_bottlebeck_features()
# train_top_model()
predict()

