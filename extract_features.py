# importing the necessary packages
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help='path to the input dataset')
ap.add_argument("-o", "--output", required=True, help='path to output HDF5 file')
ap.add_argument("-b", "--batch-size", type=int, default=32, help='batch size of images to be passes into the network')
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help='size of the feature extraction buffer')
args = vars(ap.parse_args())

bs = args['batch_size']

# Grabbing the list of images from the dataset, then randomly shuffling them for easy training and testing splits
# ... via array slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
random.shuffle(imagePaths)

# Extracting the class labels (angles) from the image paths then encode the labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load vgg network
print("[INFO] Loading network...")
model = VGG16(weights='imagenet', include_top=False)

# Initializing the HDF5 dataset writer, then store the class label names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args['output'], dataKey="features",
                            buffSize=args['buffer_size'])
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(imagePaths), bs):

    # Extract the batch of images and labels, then initialize the list of actual images and labels
    batchPaths = imagePaths[i: i + bs]
    batchLabels = labels[i: i+bs]
    batchImages = []

    # looping over the images and labels in current batch
    for (j, imagePath) in enumerate(batchPaths):

        # Loading the image using keras and resizing it to (224, 224)
        image = load_img(imagePath, target_size=(224, 224))

        image = img_to_array(image)

        # Preprocessing the image by expanding the dimensions and preprocessing it using imagenet_utils
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    # Having created batch of preprocessed images now we pass them to the model to calculate their features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # Reshaping the features so that each image is represented by a flattened feature vector of the max pooling output
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # Adding the features and labels to our HDF5 Dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# Close the datset
dataset.close()
pbar.finish()


    

















