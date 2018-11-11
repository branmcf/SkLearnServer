#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.neighbors import KNeighborsClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np
import imageio
import base64

from pdb import set_trace as bp



class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))

        vals = data['feature']
        vals = vals[0]
        fvals = imageio.imread(base64.b64decode(vals))
        fvals = fvals.flatten()
        fvals = fvals.tolist()
        # with open('test.txt','w') as f:
        #     for item in fvals:
        #         print >> f, item
        fvals = fvals[0:50]
        label = data['label']
        sess  = data['dsid']

        dbid = self.db.labeledinstances.insert(
            {"feature":fvals,"label":label,"dsid":sess}
            );
        self.write_json({"id":str(dbid),
            "feature":[str(len(fvals))+" Points Received",
                    "min of: " +str(min(fvals)),
                    "max of: " +str(max(fvals))],
            "label":label})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        if a == None:
            newSessionId = 1
        else:
            newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        dsid = self.get_int_arg("dsid",default=0)

        # create feature vectors from database
        f=[]
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            tmpList = np.array([],object)
            for val in a['feature']:
                tmpList = np.append(tmpList,float(val))
                # tmpList.np.append(float(val))

            # tmpList = np.array(tmpList).flatten().tolist()
            # tmpList = list(tmpList)
            f.append(tmpList)
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
        f = np.array(f)
        shape = f.shape[0]
        f = f.reshape(shape, 50)
        # f = f.tolist()
        # create label vector from database
        l=[];
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            l.append(a['label'])
        
        # fit the model to the data
        c1 = KNeighborsClassifier(n_neighbors=1);
        acc = -1;
        if l:
            c1.fit(f,l) # training
            lstar = c1.predict(f)
            #check 2 ses if clf has the key
            if dsid not in self.clf:
                self.clf[dsid] = c1
            #self.clf = c1 # at the dataset self.clf is now dict
                acc = sum(lstar==l)/float(len(l))
                bytes = pickle.dumps(c1)
                self.db.models.update({"dsid":dsid},
                    {  "$set": {"model":Binary(bytes)}  },
                    upsert=True)

        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":acc})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    

        vals = data['feature']
        vals = vals[0]
        fvals = imageio.imread(base64.b64decode(vals))
        fvals = fvals.flatten()
        fvals = fvals.tolist()
        fvals = fvals[0:50]
        # print fvals
        # fvals = np.array(fvals).reshape(1, -1)
        dsid  = data['dsid']
        # load requested classifier from mongodb and save in dict
        if dsid not in self.clf:
            try:
                tmp = self.db.models.find_one({"dsid":dsid})
                self.clf[dsid] = pickle.loads(tmp['model'])
            except ValueError:
                print("Oops!  We encountered an error...")
            else:
                print("Oops!  We encountered an error...")


        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        # if(self.clf == []):
        #     print('Loading Model From DB')
        #     tmp = self.db.models.find_one({"dsid":dsid})
        #     self.clf = pickle.loads(tmp['model'])
        predLabel = self.clf[dsid].predict([fvals]);
        
        self.write_json({"prediction":str(predLabel)})
