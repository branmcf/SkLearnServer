#!/usr/bin/python

# imports
from pymongo import MongoClient
import tornado.web
from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options
from basehandler import BaseHandler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np
import imageio
import base64
import cv2
import matplotlib.pyplot as plt
from pdb import set_trace as bp
import requests
import operator

# Declare and Assert Subscription Key
subscription_key = '9b5099e9f4834d3899cdb1b09a244a6c'
assert subscription_key

# Endpoint URL
face_api_url = 'https://southcentralus.api.cognitive.microsoft.com/face/v1.0/detect'


# write out to the screen which handlers were used
class PrintHandlers(BaseHandler):
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

# save data point and class label to database
class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        # get data from post request
        data = json.loads(self.request.body.decode("utf-8"))

        # get image data
        vals = data['feature']
        vals = vals[0]

        fvals = imageio.imread(base64.b64decode(vals))

        # convert RGB image to BGR for opencv and do downsampling
        cv2_img = cv2.cvtColor(fvals, cv2.COLOR_RGB2BGR)

        # downsample
        cv2_img = cv2.pyrDown(cv2_img)
        cv2_img = cv2.pyrDown(cv2_img)
        cv2_img = cv2.pyrDown(cv2_img)

        # get quarter row
        qtr_row = cv2_img.shape[0] // 2 // 2

        # get middle column
        mid_col = cv2_img.shape[1] // 2

        # get target pixel
        color_target = cv2_img[qtr_row, mid_col].tolist()
        
        # set label and session variables
        label = data['label']
        sess  = data['dsid']

        # insert into database
        dbid = self.db.labeledinstances.insert(
            {"feature":color_target,"label":label,"dsid":sess}
            );
        
        # send response to phone
        self.write_json({"id":str(dbid),
            "feature":[str(len(color_target))+" Points Received",
                    "min of: " +str(min(color_target)),
                    "max of: " +str(max(color_target))],
            "label":label})

# get a new dataset ID for building a new dataset
class RequestNewDatasetId(BaseHandler):
    def get(self):
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

# train a new model (or update) for given dataset ID
class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        dsid = self.get_int_arg("dsid",default=0)
        modelId = self.get_int_arg("modelName",default=0)
        modelId = str(modelId)

        # create feature vectors from database
        f=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            tmpList = np.array([],object)
            for val in a['feature']:
                tmpList = np.append(tmpList,float(val))
            f.append(tmpList)
        f = np.array(f)

        l=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            l.append(a['label'])
        
        # fit the model to the data
        c1 = KNeighborsClassifier(n_neighbors=3)
        if modelId == "0":
            print 'UPDATE KNN'
            c1 = KNeighborsClassifier(n_neighbors=3);
        elif modelId == "1":
            print 'UPDATE RN'
            c1 = RadiusNeighborsClassifier(radius=30.0)
        acc = -1;
        if l:
            # training
            c1.fit(f,l)
            lstar = c1.predict(f)

            #check 2 ses if clf has the key
            if dsid not in self.clf:
                print 'CLF: ', self.clf
                self.clf[dsid] = c1

            #self.clf = c1 # at the dataset self.clf is now dict
                acc = sum(lstar==l)/float(len(l))
                bytes = pickle.dumps(c1)
                self.db.models.update({"dsid":dsid},
                    {  "$set": {"model":Binary(bytes)}  },
                    upsert=True)

        # send back the resubstitution accuracy
        self.write_json({"resubAccuracy":acc})

# predict the class of a sent feature vector
class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        # get data from post request
        data = json.loads(self.request.body.decode("utf-8"))    

        # get image data
        vals = data['feature']
        vals = vals[0]

        emotion_vals = vals
        emotion_vals = base64.b64decode(emotion_vals)

        # DELETE
        # print "EMOTION: ", emotion_vals
        # cv2.imwrite("current_face.png", cv2.cvtColor(emotion_vals, cv2.COLOR_RGB2BGR))


        fvals = imageio.imread(base64.b64decode(vals))



        # set request headers and params
        headers  = {'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream" }    
        params = {
            'returnFaceId': 'false',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'emotion',
        }

        # do api call
        response = requests.post(face_api_url, params=params, headers=headers, data=emotion_vals)

        # get json from response
        emotions = response.json()
        if (len(emotions) > 0):
            emotions = emotions[0]
            emotions = emotions['faceAttributes']['emotion']
            #none, sad, surprise
            safe_emotions = {} 
            safe_emotions['angry'] = emotions['anger']
            safe_emotions['happiness'] = emotions['happiness']
            safe_emotions['neutral'] = emotions['neutral']
            safe_emotions['sad'] = emotions['sadness']
            safe_emotions['surprise'] = emotions['surprise']

            max_emotion = max(safe_emotions.iteritems(), key=operator.itemgetter(1))[0]
        else:
            print 'NO EMOTION RETURNED: ', emotions
            max_emotion = 'none'

        # convert RGB image to BGR for opencv and do downsampling
        cv2_img_p = cv2.cvtColor(fvals, cv2.COLOR_RGB2BGR)

        # downsample
        cv2_img_p = cv2.pyrDown(cv2_img_p)
        cv2_img_p = cv2.pyrDown(cv2_img_p)
        cv2_img_p = cv2.pyrDown(cv2_img_p)

        # get quarter row
        qtr_row = cv2_img_p.shape[0] // 2 // 2

        # get middle column
        mid_col = cv2_img_p.shape[1] // 2

        # get target pixel
        color_target = cv2_img_p[qtr_row, mid_col].tolist()

        # get dsid
        dsid  = data['dsid']

        # load requested classifier from mongodb and save in dict
        if dsid not in self.clf:
            try:
                tmp = self.db.models.find_one({"dsid":dsid})
                self.clf[dsid] = pickle.loads(tmp['model'])
                # print self.clf[dsid]
            except:
                print("Oops!  We encountered an error...")
            # else:
            #     print("Oops!  We encountered an uncaught error...")

        # load the model from the database (using pickle)
        predLabel = self.clf[dsid].predict([color_target]);
        predResponse = predLabel[0]
        # server prints
        print "skin prediction: ", str(predResponse)
        print "MAX EMOTION: ", max_emotion

        # write response to phone
        self.write_json({"prediction":str(predLabel), "emotion": str(max_emotion)})









# OLD CODE

# if(self.clf == []):
#     print('Loading Model From DB')
#     tmp = self.db.models.find_one({"dsid":dsid})
#     self.clf = pickle.loads(tmp['model'])

# print 'F SHAPE', f.shape
# shape_val = f.shape[0]*f.shape[1]
# f = f.reshape(1, shape_val)
# f = f.tolist()
# create label vector from database

# tmpList.np.append(float(val))

# tmpList = np.array(tmpList).flatten().tolist()
# tmpList = list(tmpList)

# print cv2_img
# print 'TYPE: ', type(cv2_img)
# print 'SHAPE: ', cv2_img.shape
# flatten the array and listify 
# cv2_img = cv2_img.flatten()

# cv2_img = cv2_img.tolist()

# print the imagee
# print len(cv2_img)


# fvals = fvals.flatten()
# fvals = fvals.tolist()
# fvals = fvals[0:4000]

# convert pic to rgb for plotting
# RGB_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

# print pic
# write the image and show it
# cv2.imwrite("postsample.jpg", RGB_img)
# plt.figure()
# plt.imshow(RGB_img)
# plt.show(block=True)


# print f
# bp()
# for val in a['feature']:
#     for sublist in val:
#         for e in sublist:
#             tmpList.append(float(e))
# f.append(tmpList)
    # val = list(val)
        # f.append(float(i))

# f.append([float(val) for val in a['feature']])
# print np.array(f).shape\