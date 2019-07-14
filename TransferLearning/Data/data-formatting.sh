#!/usr/bin/env bash
# Loop through all the files in the directory and run the med2image program over it.
# This will convert all the files in the directory to JPG

# Creates an 80/20 split between training and validation
# 31 training MRI and


# Creates the following directory structure

#* Training
#			* Bipolar
#					* CHRM2049-bipolar-slice000.jpg
#						…
#                   * CHRM2049-bipolar-slice179.jpg
#
#	                   ...

#					* CHRM2023-control-slice000.jpg
#						…
#                   * CHRM2023-control-slice179.jpg
#* Validation
#           * Bipolar
#					* CHRM2025-control-slice000.jpg
#						…
#                   * CHRM2025-control-slice179.jpg
#
#	                  ...

#					* CHRM2025-control-slice000.jpg
#						…
#                   * CHRM2025-control-slice179.jpg
#           * Control
#					* CHRM2098-control-slice000.jpg
#						…
#                   * CHRM2098-control-slice179.jpg
#
#	                ...
#
#					* CHRM2099-control-slice000.jpg
#						…
#                   * CHRM2099-control-slice179.jpg

# A counter to direct the files to either train or validation folders
COUNTER=0
# Whether or not it's control or bipolar
DATA_TYPE=$1
# Path to input folder
FOLDER=$2

BASE_PATH=$PWD

NAME_COUNTER=0

# Loop to run each file from the input folder into the med2image converter, go from NII to jpeg.
for filename in ${FOLDER}/*.nii; do

    # See whether to go to train or validation
    if (( $COUNTER >= 10)); then
        JPEG_OUTPUT_FOLDER="converted/train/$DATA_TYPE/"
    else
        JPEG_OUTPUT_FOLDER="converted/validation/$DATA_TYPE/"
    fi

    # Cut the unwanted bits of the input path to just get a meaningful name
#    FILENAME=$(echo ${filename}| cut  -d'_' -f 1 | cut -d'/' -f4)
     FILENAME="MRI-$NAME_COUNTER"
    # Extract just the brain fro the MRI
    deepbrain-extractor -i "$filename" -o tmp/${FILENAME}
    # The converter from NII to JPEG
    med2image -i "tmp/$FILENAME/brain.nii" -d "$JPEG_OUTPUT_FOLDER/tmp" -o ${FILENAME}-${DATA_TYPE}.jpg -s -1
    # This will pick the best slices from the MRI based on image entropy, then crop and rotate.
    python entropy.py -d "$BASE_PATH/$JPEG_OUTPUT_FOLDER/tmp/" -o "$JPEG_OUTPUT_FOLDER" -t ${DATA_TYPE}

    # Update the counter
   ((COUNTER++))
   ((NAME_COUNTER++))
done

rm -r ${BASE_PATH}/converted/train/${DATA_TYPE}/tmp
rm -r ${BASE_PATH}/converted/validation/${DATA_TYPE}/tmp