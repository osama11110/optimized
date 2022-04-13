# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:41:02 2022

@author: salma asif
"""

#torch & torchvision
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader


import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


#sklearn for splitting dataset and scoring functions
from sklearn.metrics import accuracy_score,roc_auc_score, precision_recall_fscore_support, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet as EffNet 

######
from skimage import exposure
#####


#Other
import matplotlib.pyplot as plt
from time import perf_counter
from scipy import stats
from PIL import Image
import pandas as pd
import numpy as np
import random
import pickle
import json
import os
import cv2 as cv

from flask import Flask
from flask import request
import json 
import wget
import shutil
import pydicom
 


app = Flask(__name__) 
@app.route('/checkpic', methods = ['POST']) 
def check_pic():

    #HyperParameters and Paths for Train & Test Dataset.
    BATCH_SIZE = 32
    #NUM_WORKERS = 8
    NUM_WORKERS = 0
    GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #TEST_DIR = './CXR_Covid-19_Challenge/test_set'
    #TEST_DIR = '/content/drive/MyDrive/CXR-Grand-Challenge-2021-main-CBRL/chexpert'
    #TEST_DIR = 'E:/cxr/test'
    dir_path=tempfile.mkdtemp()
    url=request.get_json()
    wget.download(url , out=(dir_path))
    
    
    def read_xray(path, voi_lut = True, fix_monochrome = True):
        dicom = pydicom.read_file(path)
        
        # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
            #print("voi lut",data)
        else:
            data = dicom.pixel_array
            #print(data)
                   
        # depending on this value, X-ray may look inverted - fix that:
        #print(fix_monochrome, "and", dicom.PhotometricInterpretation)
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
            
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint16)
        return data
    
    
    
    TEST_DIR = dir_path
    #TEST_DIR = 'https://patientdicoms.s3.us-east-2.amazonaws.com/angular2-logo.png'
    
    
    
    def getModel(Name, OutFeatures):
        if Name == "AlexNet":
            model = torchvision.models.alexnet()
            model.classifier[6] = nn.Linear(4096,OutFeatures,bias=True)
            model = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                model
            )
        elif Name=="VGG16":
            model = torchvision.models.vgg16( )
            model.classifier[6] = nn.Linear(4096,OutFeatures,bias=True)
        elif Name=="resnet18":
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(512,OutFeatures,bias=True)
            model = nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                model
            )
        elif Name=="effnet":
            model = EffNet.from_name("efficientnet-b0", in_channels=3)
            model._fc = nn.Linear(in_features=1280,out_features=OutFeatures)
        elif Name=="unet":
            json_file = open("./CXR_Covid-19_Challenge/model.json","r")
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights("./CXR_Covid-19_Challenge/unet_lung_seg.hdf5")
        return model
    from skimage import img_as_float   
    class Normalization(object):
        """Convert image to negative.
    
        Args:
            
    
        Returns:
            PIL Image: Negative version of the input.
             
        """
        def __init__(self):
            pass
        
        def __call__(self, img, key="hist"):
            """
            Args:
                img (PIL Image): Image to be converted to Negative.
    
            Returns:
                PIL Image: Negatived image.
            """
            
            
            if key=="hist":
                img = exposure.equalize_hist(img_as_float(img))
                img = np.array(img, dtype=np.float32)
                
            return img
        
        
    
    class CheckPoint():
        def __init__(self,Parameters):
            self.Count = 0
            self.BestLoss = float('inf')
            self.BestEpoch = -1
            
            self.Patience = Parameters["Patience"]
            self.Path = Parameters["SavePath"]
            self.earlyStopping=Parameters["earlyStopping"]
    
        def check(self,epoch,loss):
            torch.save({"Model":self.Model.state_dict(),"Optimizer":self.Optimizer.state_dict()},self.Path+"/Model.pth")
            if loss>self.BestLoss:
                self.Count+=1
            else:
                self.Count=0
                self.BestLoss = loss
                self.BestEpoch = epoch
                torch.save({"Model":self.Model.state_dict(),"Optimizer":self.Optimizer.state_dict()},self.Path+"/BestModel.pth")
           # with open(self.Path+"/Results.txt", 'wb') as file:
                #pickle.dump(self.Results,file)
            if self.earlyStopping:
                if self.Count==self.Patience:
                    print("\nEarly Stopping!")
                    print("Model didn't improved for",self.Patience,"epochs.")
                    return False
                return True
            else:
                return True
            
            
            
    class XRayDataset(torch.utils.data.Dataset):
        """Face Landmarks dataset."""
    
        def __init__(self, files, root_dir, transform=None):
            self.files = files
            self.root_dir = root_dir
            self.transform = transform
    
        def __len__(self):
            return len(self.files)
    
        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.files[idx])
            image=read_xray(img_name)
            #image = cv.imread(img_name)
            image=np.uint8(image)
            image=cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)
            sample = {'case': self.files[idx], 'image': image}
            return sample
        
        
        
    class Hashtag(CheckPoint):
        def __init__(self, Parameters):
            CheckPoint.__init__(self,Parameters["CheckPoint"])
            self.Model = Parameters["Model"]
            self.Criterion = Parameters["Criterion"]
            self.Optimizer = Parameters["Optimizer"]
            self.Scheduler = Parameters["Scheduler"]
            self.TrainLoader = Parameters["TrainLoader"]
            self.ValidateLoader = Parameters["ValidateLoader"]
            self.Labels = Parameters["Labels"]
            self.Device = Parameters["Device"]
            self.Results = Parameters["Results"]
            self.Softmax = nn.Softmax(dim=1)
            self.ontoDevice()
    
        def ontoDevice(self):
            self.Model.to(DEVICE)
    
        def predict(self,TestLoader):
            self.Model.eval()
            Results = []
            with torch.no_grad():
                for batch,data in enumerate(TestLoader):
                    case,images = data
                    case = data["case"]
                    images = data["image"]
                    images = images.to(DEVICE)
                    pred = self.Softmax(self.Model(images).to("cpu"))
    #                 pred[:,0] = pred[:,0] **2.6
    #                 pred[:,1] = pred[:,1] **3.6
    #                 pred[:,2] = pred[:,2] **0.9
                    pred = pred.argmax(1)
                    for i in range(len(case)):
                        Results.append(int(pred[i]))
            return Results
    
    
    #Ensembling
    Voting = []
    img_size = 256
    for FoldID in range(5):
    #    path = "./EffNet-100-05/Model.pth"
        path = 'BestModel.pth'
        transform_test = transforms.Compose([
                                                transforms.Resize((img_size,img_size)),
                                                #Normalization(),
                                                transforms.ToTensor(),
            
                                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        test_data_retriever = XRayDataset(sorted(os.listdir(TEST_DIR)),TEST_DIR,transform = transform_test)
        test_data_loader = DataLoader(
                            test_data_retriever,
                            batch_size = BATCH_SIZE,
                            shuffle = True,
                            num_workers = NUM_WORKERS,
                            pin_memory = True
        )
        model = getModel("effnet",3).to(DEVICE)
        #model.load_state_dict(torch.load(path)["Model"])
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))["Model"])
        
        criterion = nn.CrossEntropyLoss()
    
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        Params = {
            "Model":model,
            "Criterion":criterion,
            "Optimizer":optimizer,
            "Scheduler":scheduler,
            "TrainLoader":None,
            "ValidateLoader":test_data_loader,
            "Labels":None,
            "Device":DEVICE,
            "Results":[],
            "CheckPoint":{
                        "Patience":5,
                        "SavePath":path,
                        "earlyStopping":False
                        }
                 }
        hashtag = Hashtag(Params)
        Results = hashtag.predict(test_data_loader)
        Finish = perf_counter()
    
    
    print(Results)
    
    return json.dumps({"result":Results})

if __name__ == "__main__": 
    app.run(port=5000)

