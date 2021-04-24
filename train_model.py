# Using the features extracted by the VGG Net and inputing it into logistic regression classifier to classify the orientation of the image
# importing the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

# Constructing the argument parser and parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-db", "--dataset", required=True, help='path to the feature HDF5 database')
ap.add_argument("-m", "--model", required=True, help='path to the output model')
ap.add_argument("-j", "--jobs", type=int, default=-1, help='# of jobs to run when tuning hyperparameters')
args = vars(ap.parse_args())

# Loading the hdf5 file that contains the features, and spliting the index of training and testing data
db = h5py.File(args['dataset'], "r")
i = int(db['labels'].shape[0] * 0.75)

# Defing the set of parameters that we want to tune : C - inverse of regularization
print("[INFO] Tuning Hyperparameters...")
params = {"C": [0.01, 0.1, 1.0, 10.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args['jobs'])
model.fit(db["features"][:i], db['labels'][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# Evaluate the model
print("[INFO] Evaluating...")
preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], preds, target_names=db['label_names']))

# Serializing the model to the disk
print("[INFO] saving model...")
f = open(args['model'], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()





















