import os
from src.predictor.predict import predict
from src.predictor.image.prepDataIO import prep_data


# Test that prep_data stores JPG files in the tmp directory
nii_test_file = "/Users/pmartyn/PycharmProjects/Thesis/test/VGG16/test-file/sub-60001_anat_sub-60001_T1w.nii"
nii_base_out_path = "/Users/pmartyn/PycharmProjects/Thesis/test/VGG16/tmp/12345/"
nii_output_dir = nii_base_out_path + "proc_img"


def number_of_files():
    return len(os.listdir(nii_output_dir))


if not os.path.exists(nii_base_out_path):
    os.makedirs(nii_base_out_path)

prep_data(nii_test_file,
          nii_base_out_path)
#
# print("Number of files ", number_of_files())

# Check 6 JPEG were saved to the tmp directory.
assert number_of_files() == 10
assert all(file.endswith('.jpg') for file in os.listdir(nii_output_dir))

try:
    image_data = predict(nii_base_out_path)
    print(image_data)
    assert all(file["class-label"] == 'Bipolar' for file in image_data)
    assert all(file["filename"] is not None for file in image_data)
    assert all(file["class-probability"] is not None for file in image_data)
except ValueError as e:
    print("Cannot do prediction: ", e)



