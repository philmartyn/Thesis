from med2image import med2image
from deepbrain import Extractor
import nibabel as nib
import os
from src.VGG16.image.helpers import get_image_entropies, crop_and_save_image

# Taken from https://github.com/iitzco/deepbrain/blob/master/bin/deepbrain-extractor
# Changed the code a little because I didn't want the brain mask image saved.


def prep_data(filename_path):

    nii_output_dir = os.getenv('TMP_DIR_PATH', "/Users/pmartyn/PycharmProjects/Thesis/tmp/")

    if not os.path.exists(nii_output_dir):
        os.makedirs(nii_output_dir)

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
    nib.save(brain, os.path.join(nii_output_dir, "brain.nii"))

    jpg_dir_path = nii_output_dir + '/jpg/'

    if not os.path.exists(jpg_dir_path):
        os.makedirs(jpg_dir_path)

    med2image.med2image_nii(
        inputFile=nii_output_dir + "/brain.nii",
        outputDir=jpg_dir_path,
        outputFileStem='NII-file',
        outputFileType="jpg").run()

    processed_img_dir = nii_output_dir + "/proc_img/"

    if not os.path.exists(processed_img_dir):
        os.makedirs(processed_img_dir)

    image_entropies = get_image_entropies(jpg_dir_path)
    crop_and_save_image(image_entropies, jpg_dir_path, processed_img_dir)
