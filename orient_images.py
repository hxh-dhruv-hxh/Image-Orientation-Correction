# import the necessary packages
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import h5py
import cv2

# Constructing the argument parser and parsing the images
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help='path to HDF5 database')
ap.add_argument("-i", "--dataset", required=True, help='path to the input image dataset')
ap.add_argument("-m", "--model", required=True, help='path to trained orientation model')
args = vars(ap.parse_args())

# loading the label name from the HDF5 dataset
db = h5py.File(args['db'])
labelNames = [int(angle) for angle in db['label_names'][:]]
db.close()

# Grab the paths to the testing images and randomly sample them
print("[INFO] sampling images...")
imagePaths = list(paths.list_images(args['dataset']))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# load the VGG network
print("[INFO] loading network...")
vgg = VGG16(weights='imagenet', include_top=False)

# Load the orientation model
print("[INFO] loading model...")
model = pickle.loads(open(args['model'], "rb").read())

# Loop over the image paths
for imagePath in imagePaths:

    # loading the image using openCV so we can maipulate it after classification
    orig = cv2.imread(imagePath)

    # preprocessing the image using the keras helper utilities
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # preprocess the image by expanding its dimensions and substracting the RGB pixel intensity from the imagenet dataset
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # Passing the image to obtain its feature vector
    features = vgg.predict(image)
    features = features.reshape((features.shape[0], 512*7*7))

    # Now we will pass these CNN features to our classifier to obtain the results
    angle = model.predict(features)
    angle = labelNames[angle[0]]

    # Now that the model has predicted orientation of the image we can correct its angle by substracting it with 360
    rotated = imutils.rotate_bound(orig, 360-angle)

    # Displaying the original and corrected image
    cv2.imshow("original", orig)
    cv2.imshow("Corrected", rotated)
    cv2.waitKey(0)















