import cv2

class MeanPreprocessor:

    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):

        (B, G, R) = cv2.split(image.astype('float'))

        # Substracting mean from each channel from the image
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # Merging the channels back and returning the image
        return cv2.merge([B, G, R])

