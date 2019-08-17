# Borrowed from Francois Chollet blog - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras import applications, Input, Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Flatten, Dense
import numpy as np

weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/bottleneck_weights.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/TransferLearning/Data/Crossfolds/Fold1-Orig/converted/train'
validation_data_dir = '/Users/pmartyn/PycharmProjects/Thesis/TransferLearning/Data/Crossfolds/Fold1-Orig/converted/validation'
nb_train_samples = 4520
nb_validation_samples = 1280
epochs = 50
batch_size = 16

# build the VGG16 network
vgg16 = applications.VGG16(weights='imagenet', include_top=False, input_shape = (img_width, img_width, 3))
print('Model loaded.')

train_data = np.load(open('bottleneck_features_train.npy', 'rb'))

dense_input  = Input(shape=train_data.shape[1:])
dense_output = Flatten(name='flatten')(dense_input)
dense_output = Dense(256, activation='relu', name='fc2')(dense_output)
dense_output = Dense(1, activation='sigmoid', name='predictions')(dense_output)

top_model = Model(inputs=dense_input, outputs=dense_output, name='top_model')
top_model.load_weights(top_model_weights_path)

top_model.summary()

block5_pool = vgg16.get_layer('block5_pool').output

full_output = top_model(block5_pool)
full_model = Model(inputs=vgg16.input, outputs=full_output)


full_model.summary()


for layer in full_model.layers[:15]:
    layer.trainable = False

full_model.compile(loss='binary_crossentropy',
                   optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                   metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
full_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

output_weights_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/fine-tune-weights.h5'
output_model_path = '/Users/pmartyn/PycharmProjects/Thesis/Output/fine-tune-model.h5'
full_model.save_weights(output_weights_path)
full_model.save(output_model_path)
