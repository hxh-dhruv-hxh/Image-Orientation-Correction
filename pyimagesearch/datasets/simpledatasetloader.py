# Code Objective : To build a simple dataset loader

import numpy as np
import cv2
import os

class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):

        self.preprocessors = preprocessors

        # If the preprocessors passed are None then initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):

        data = []
        labels = []

        # Looping over the input images
        # Assuming that the path has the following format: /path/to/dataset/{class}/{image}.jpg
        for (i, imagePath) in enumerate(imagePaths):

            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # Check to see if the preprocessors are None or Not
            if self.preprocessors is not None:
                # If not then we will loop over them and apply each one of them on the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # Now, that we have applied the preprocessors the data is treated as a feature vector is ready to be stored in the variables
            data.append(image)
            labels.append(label)

            # To show an update to data being processed
            if verbose > 0  and i > 0 and (i+1) % verbose == 0:
                print("[INFO] Processed {}/{}".format(i+1, len(imagePaths)))

        return (np.array(data), np.array(labels))




