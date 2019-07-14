import os
import shutil
from keras.models import load_model
from keras.applications import VGG16
import cv2

img_width, img_height = 150, 150


def predict():

    img_base_path = os.getenv('TMP_DIR_PATH', "/Users/pmartyn/PycharmProjects/Thesis/tmp/")
    print(img_base_path)
    img_input_path = img_base_path + "proc_img/"

    if os.path.exists(img_input_path) and os.listdir(img_input_path):

        model = load_model(os.getenv('MODEL_PATH', "/Users/pmartyn/PycharmProjects/Thesis/src/VGG16/static/bottleneck_model.h5"))
        other_model = VGG16(include_top=False, weights='imagenet')

        images_data = []

        for img in os.listdir(img_input_path):
            print("\n" + img_input_path + img)
            print(img)

            image = cv2.imread(img_input_path + img)
            image = cv2.resize(image, (img_width, img_width))

            # scale the pixel values to [0, 1]
            image = image.astype("float") / 255.0
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

            features = other_model.predict(image)
            preds = model.predict_classes(features)[0][0]
            preds_num = model.predict(features)[0][0]

            print("Prediction Class: ", preds)
            print("Predict Probabilities: ", preds_num)

            if preds == 0:
                print("Label: Bipolar")
                label = "Bipolar"
            else:
                print("Label : Control")
                label = "Control"

            dictionary = {"filename": img,
                          "class-label": label,
                          "class-probability": preds_num * 100}

            print("\nclass-probability ", preds_num)
            images_data.append(dictionary)

        try:
            print("Removing tmp directory.")
            shutil.rmtree(img_base_path)
        except Exception as e:
            print("Can't delete tmp! " + str(e))

        return images_data

    else:
        raise ValueError('Images required for prediction not found.')

