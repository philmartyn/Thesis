import os
from src.predictor.predict import predict
from src.predictor.image.prepDataIO import prep_data


# Test that prep_data stores JPG files in the tmp directory
nii_output_dir = "/Users/pmartyn/PycharmProjects/Thesis/tmp/proc_img"


def number_of_files():
    return len(os.listdir(nii_output_dir))


# assert number_of_files() == 0

prep_data("/Users/pmartyn/PycharmProjects/Thesis/TransferLearning/Data/prediction-data/MRI/Bipolar/CHRM2053_MPRAGE_T13D.nii")
#
# print("Number of files ", number_of_files())

# Check 6 JPEG were saved to the tmp directory.
assert number_of_files() == 10
assert all(file.endswith('.jpg') for file in os.listdir(nii_output_dir))

try:
    image_data = predict()
    print(image_data)
    assert all(file["class-label"] == 'Bipolar' for file in image_data)
    assert all(file["filename"] is not None for file in image_data)
    assert all(file["class-probability"] is not None for file in image_data)
except ValueError as e:
    print("Cannot do prediction: ", e)



