# import the necessary packages
from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import cv2
import os

# constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help='path to the input directory of the images')
ap.add_argument("-o", "--output", required=True, help='path to output directory of rotated images')
args = vars(ap.parse_args())

# Grabing the images using the imagePaths(limiting to only 10,000) and shuffling them to create training and testing splits
imagePaths = list(paths.list_images(args['dataset']))[:10000]

# initializing the dictionary to keep track of number of images of each angle
angles = {}

# initlializing the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# looping over the imagepaths
for (i, imagePath) in enumerate(imagePaths):

    # Determining the rotation angle and loading the image
    angle = np.random.choice([0, 90, 180, 270])
    image = cv2.imread(imagePath)

    # If there is issue loading the image from disk then we skip it
    if image is None:
        continue

    # rotate the images to the selected angle
    image = imutils.rotate_bound(image, angle)
    base = os.path.sep.join([args['output'], str(angle)])

    # if the base path does not exist already, create it
    if not os.path.exists(base):
        os.makedirs(base)

    # Extracting the image extension, and constructing the full path to the output file
    ext = imagePath[imagePath.rfind("."):]
    outputPath = [base, "image_{}{}".format(str(angles.get(angle, 0)).zfill(5), ext)]
    outputPath = os.path.sep.join(outputPath)

    # Save the image
    cv2.imwrite(outputPath, image)

    # Update the angle count
    c = angles.get(angle, 0)
    angles[angle] = c + 1
    pbar.update(i)

# Finish the progress bar
pbar.finish()

# Looping over the angles and displaying its counts
for angle in sorted(angles.keys()):
    print("[INFO] angle={}: {:,}".format(angle, angles[angle]))



























