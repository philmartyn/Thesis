from med2image import med2image
from deepbrain import Extractor
import nibabel as nib
import os
from Data import helpers

# Taken from https://github.com/iitzco/deepbrain/blob/master/bin/deepbrain-extractor
# Changed the code a little because I didn't want the brain mask image saved.


def prep_data(filename):

    input_file = "/Users/pmartyn/Documents/College_Work/Thesis/Thesis-Frontend/thesis-frontend/resources/public/" + filename
    nii_output_dir = "/Users/pmartyn/PycharmProjects/Thesis/tmp"

    if not os.path.exists(nii_output_dir):
        os.makedirs(nii_output_dir)

    p = 0.5
    img = nib.load(input_file)

    affine = img.affine
    img = img.get_fdata()

    extractor = Extractor()

    prob = extractor.run(img)
    mask = prob > p

    brain = img[:]
    brain[~mask] = 0
    brain = nib.Nifti1Image(brain, affine)
    nib.save(brain, os.path.join(nii_output_dir, "brain.nii"))

    jpg_output_dir_path = nii_output_dir + '/jpg/'

    if not os.path.exists(jpg_output_dir_path):
        os.makedirs(jpg_output_dir_path)

    med2image.med2image_nii(
        inputFile=nii_output_dir + "/brain.nii",
        outputDir=jpg_output_dir_path,
        outputFileStem='output-file',
        outputFileType="jpg").run()

    prepped_img_dir = nii_output_dir + "/proc_img/"

    if not os.path.exists(prepped_img_dir):
        os.makedirs(prepped_img_dir)

    image_entropies = helpers.get_image_entropies(jpg_output_dir_path)
    helpers.crop_image(image_entropies, jpg_output_dir_path, prepped_img_dir)