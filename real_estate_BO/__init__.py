import logging
import pickle
import azure.functions as func
import pandas as pd
import joblib
import json
import numpy as np
from azure.storage.blob import BlobServiceClient, BlobClient

def main(req: func.HttpRequest,
         
         output: func.Out[bytes]) -> func.HttpResponse:


    # load the model
    model = joblib.load('linear_reg.pkl')

    # gather new sample
    age = float(req.params['Age'])
    dist = float(req.params['Dist'])
    num_stores = float(req.params['Num_stores'])
    lat = float(req.params['Lat'])
    long_ = float(req.params['Long'])

    X_new = [[age,dist,num_stores,lat,long_]]

    
    # score on the new sample
    pred_1 = model.predict(X_new)
    pred_1 = int(pred_1)

    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError
    
    pred = json.dumps(pred_1, default=set_default)
    
    #print(pred)
    output.set(str(pred))

    return func.HttpResponse(f"Input JSON: {pred}")

    