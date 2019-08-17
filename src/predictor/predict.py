import os
import shutil
from keras.models import load_model
from keras.applications import VGG16
import cv2

img_width, img_height = 150, 150


def predict(nii_dir_path):

    print(nii_dir_path)
    img_input_path = nii_dir_path + "proc_img/"

    # Check if files are there to predict upon.
    if os.path.exists(img_input_path) and os.listdir(img_input_path):

        # Load the models needed for prediction
        model = load_model(os.getenv('MODEL_PATH',
                                     "/Users/pmartyn/PycharmProjects/Thesis/src/Predictor/static/bottleneck_model.h5"))
        base_model = VGG16(include_top=False, weights='imagenet')

        images_data = []

        # Predict against each image in the image directory.
        for img in os.listdir(img_input_path):
            print("\n" + img_input_path + img)
            print(img)

            # Load the images, resize, scale and reshape
            image = cv2.imread(img_input_path + img)
            image = cv2.resize(image, (img_width, img_width))

            # scale the pixel values to [0, 1]
            image = image.astype("float") / 255.0
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

            # Get the features of the image using the VGG16 model
            features = base_model.predict(image)
            # And then predict against the trained top model.
            preds = model.predict_classes(features)[0][0]
            preds_num = model.predict(features)[0][0]

            print("Prediction Class: ", preds)
            print("Predict Probabilities: ", preds_num)

            # Get the label value
            if preds == 0:
                print("Label: Bipolar")
                label = "Bipolar"
            else:
                print("Label : Control")
                label = "Control"

            # Store in a dictionary
            dictionary = {"filename": img,
                          "class-label": label,
                          "class-probability": preds_num * 100}

            print("\nclass-probability ", preds_num)
            images_data.append(dictionary)

        # Delete the tmp directory. Important to not leave confidential data lying around.
        try:
            print("Removing tmp directory.")
            shutil.rmtree(nii_dir_path)
        except Exception as e:
            print("Can't delete tmp! " + str(e))

        return images_data

    else:
        raise ValueError('Images required for prediction not found.')

