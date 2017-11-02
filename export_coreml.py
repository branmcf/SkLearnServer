#!/usr/bin/python
'''Read from PyMongo, make simply model and export for CoreML'''

# make this work nice when support for python 3 releases
from __future__ import print_function

# database imports
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

# model imports
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# export 
import coremltools


dsid = 3
client  = MongoClient(serverSelectionTimeoutMS=50)
db = client.sklearndatabase


# create feature vectors from database
X=[];
for a in db.labeledinstances.find({"dsid":dsid}): 
    X.append([float(val) for val in a['feature']])

# create label vector from database
y=[];
for a in db.labeledinstances.find({"dsid":dsid}): 
    y.append(a['label'])


print("Found",len(y),"labels and",len(X),"feature vectors")
print("Unique classes found:",np.unique(y))

clf = RandomForestClassifier(n_estimators=50)
print("Training Model", clf)

clf.fit(X,y)

print("Exporting to CoreML")

coreml_model = coremltools.converters.sklearn.convert(
	clf) 

# save out as a file
coreml_model.save('RandomForestAccel.mlmodel')

client.close() 