import json
import os
import shutil
from keras.models import load_model
from keras.applications import VGG16
import cv2
import numpy as np
from keras.preprocessing import image

img_width, img_height = 150, 150


def predict():

    img_input_dir = "/Users/pmartyn/PycharmProjects/Thesis/tmp/proc_img"
    print("In predicter")

    if os.path.exists(img_input_dir) and os.listdir(img_input_dir):
        model = load_model('/Users/pmartyn/PycharmProjects/Thesis/src/VGG16/bottleneck_model.h5')
        other_model = VGG16(include_top=False, weights='imagenet')

        images_data = []

        for img in os.listdir(img_input_dir):
            print("\n" + img_input_dir + "/" + img)
            print(img)
            image = cv2.imread(img_input_dir + "/" + img)
            image = cv2.resize(image, (img_width, img_width))

            # scale the pixel values to [0, 1]
            image = image.astype("float") / 255.0

            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

            features = other_model.predict(image)
            preds = model.predict_classes(features)[0][0]
            preds_num = model.predict(features)[0][0]

            print("Prediction Class: ", preds)
            print("Predict Probalities: ", preds_num)

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

        print("\nImage Data ", images_data)
        print("\nImage Data JSON ", json.dumps(images_data, ensure_ascii=False))

        for file in os.listdir("/Users/pmartyn/PycharmProjects/Thesis/tmp/"):
            try:
                print("Removing file.", file)
                os.remove("/Users/pmartyn/PycharmProjects/Thesis/tmp/" + file)
            except Exception as e:
                shutil.rmtree("/Users/pmartyn/PycharmProjects/Thesis/tmp/" + file)

        return images_data

    else: print("Doesn't predict")


    # for img in os.listdir(image_dir):
    #     print(img)
    #     img = image.load_img(image_dir + "/" + img, target_size=(img_width, img_height))
    #     img = image.img_to_array(img)
    #     img = np.expand_dims(img, axis=0)
    #     images.append(img)
    #
    # print(len(images))
    # images = np.vstack(images)
    #
    # features = other_model.predict(images)
    # preds = model.predict(features)
    #
    # print("Preds ==", preds)
    # print("Preds 1", preds[0][0])
    # print("Preds 2", preds[1][0])
    #
    # labels = preds.argmax(axis=1)
    # print("Labels", labels)
    # label_name = []
    # for label in labels:
    #     print("[INFO] Label...", label)
    #     if label == 0:
    #         label_name.append("bipolar")
    #     else:
    #         label_name.append("control")
    #
    # print("[INFO] Label...", label_name)
    # print("[INFO] Number...", preds[0][0] * 100)
    # print("[INFO] Label...", i)
    # print("[INFO] Label...", preds[0][i] * 100)
    # print("[INFO] Label...", preds)
    # print preds
    # print preds[0]
    # print (preds[0][0])

# def predict_2():
#     model = load_model('/Users/pmartyn/PycharmProjects/Thesis/src/VGG16/bottleneck_model.h5')
#     other_model = VGG16(include_top=False, weights='imagenet')
#
#     image = cv2.imread('/Users/pmartyn/PycharmProjects/Thesis/Data/test/tmp/jpg/a/3.jpg')
#     output = image.copy()
#     image = cv2.resize(image, (img_width, img_width))
#
#     # scale the pixel values to [0, 1]
#     image = image.astype("float") / 255.0
#
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#
#     features = other_model.predict(image)
#     preds = model.predict_classes(features)
#
#     i = preds.argmax(axis=1)[0]
#     # print("[INFO] Label...", i)
#     # print("[INFO] Label...", preds[0][i] * 100)
#     # print("[INFO] Label...", preds)
#     # print preds
#     # print preds[0]
#     print (preds[0][0])

# predict()
# predict_2()
