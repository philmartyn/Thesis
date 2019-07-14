# Borrowed heavily from https://github.com/marciahon29/Ryerson_MRP
# and https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
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
# img_width, img_height = 200, 235

# Paths to model output and data
top_model_weights_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/bottleneck_weights.h5'
model_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/bipolar.model'
train_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/converted/train'
validation_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/Data/converted/validation'
# Hyperparameters inputs relating to data
nb_train_samples = 4544
nb_validation_samples = 1280
init_lr = 1e-3
epochs = 100
batch_size = 40
data_array = []
labels_array = []


# Create and save feature arrays
# Bottleneck features in VGG16 model refers to the last activation maps before the fully connected layer.
def gen_and_save_bottleneck_features():
    imagegen = ImageDataGenerator(rescale=1. / 255 ) #zca_whitening=True

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print("Model created. ", model)

    generator = imagegen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True)
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
        shuffle=True)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


# Train the VGG16 fully connected layer using the data.
def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        ["bipolar"] * (nb_train_samples / 2) + ["control"] * (nb_train_samples / 2))

    print("Training data. ", train_data.size)
    print("Training labels. ", train_labels.size)
    print("Training labels. ", train_labels)

    lb = LabelBinarizer()
    training_labels = lb.fit_transform(train_labels)

    print("Training labels post labeliser. ", training_labels.size)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        ["bipolar"] * (nb_validation_samples / 2) + ["control"] * (nb_validation_samples / 2))

    val_labels = lb.fit_transform(validation_labels)

    print("Validation shape ", validation_data.shape[1])

    # Output the pickle file for use later in prediction
    f = open("/Users/pmartyn/PycharmProjects/Thesis/Output/bipolar_lb.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()

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

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.000001,)# decay=init_lr / epochs

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])


    # train_data_normalised = normalize(train_data)
    # validation_data_normalised = normalize(validation_data)

    history = model.fit(train_data, training_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, val_labels))
    model.save_weights(top_model_weights_path)
    model.save(model_path)

    with open('/Users/pmartyn/PycharmProjects/Thesis/Output/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


gen_and_save_bottleneck_features()
train_top_model()


# plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# N = epochs
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper left")
# plt.savefig(args["plot"])