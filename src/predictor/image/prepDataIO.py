from med2image import med2image
from deepbrain import Extractor
import nibabel as nib
import os
from src.predictor.image.helpers import get_image_entropies, crop_and_save_image

# Taken from https://github.com/iitzco/deepbrain/blob/master/bin/deepbrain-extractor
# Changed the code a little because I didn't want the brain mask image saved.


def prep_data(filename_path, nii_dir_path):

    print("Extracting brain from NII file.")
    # Brain extraction stuff happens here.
    p = 0.5
    img = nib.load(filename_path)

    affine = img.affine
    img = img.get_fdata()

    extractor = Extractor()

    prob = extractor.run(img)
    mask = prob > p

    brain = img[:]
    brain[~mask] = 0
    brain = nib.Nifti1Image(brain, affine)
    nib.save(brain, os.path.join(nii_dir_path, "brain.nii"))

    # Split the extracted brain NII into individual JPGs.
    jpg_dir_path = nii_dir_path + '/jpg/'

    # Create the necessary output directories
    if not os.path.exists(jpg_dir_path):
        os.makedirs(jpg_dir_path)

    # Split the NII into JPGs
    med2image.med2image_nii(
        inputFile=nii_dir_path + "/brain.nii",
        outputDir=jpg_dir_path,
        outputFileStem='NII-file',
        outputFileType="jpg").run()

    processed_img_dir = nii_dir_path + "/proc_img/"

    if not os.path.exists(processed_img_dir):
        os.makedirs(processed_img_dir)

    # Get the entropy values of each image...
    image_entropies = get_image_entropies(jpg_dir_path)
    # ...and save the 10 images with the highest entropy
    crop_and_save_image(image_entropies, jpg_dir_path, processed_img_dir)
