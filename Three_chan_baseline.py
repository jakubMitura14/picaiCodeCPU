
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import time
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from monai.utils import set_determinism
import math
import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from monai.networks.layers.factories import Act, Norm
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from comet_ml import Optimizer
import functools
import multiprocessing as mp
import os
import os.path
monai.utils.set_determinism()
from functools import partial
from os import path as pathOs

import importlib.util
import sys
import ThreeChanNoExperiment

percentSplit=0.8

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

# transformsForMain =loadLib("transformsForMain", "/mnt/disks/sdb/piCaiCode/preprocessing/transformsForMain.py")
# manageMetaData =loadLib("ManageMetadata", "/mnt/disks/sdb/piCaiCode/preprocessing/ManageMetadata.py")
# dataUtils =loadLib("dataUtils", "/mnt/disks/sdb/piCaiCode/dataManag/utils/dataUtils.py")

unets =loadLib("unets", "/mnt/disks/sdb/piCaiCode/model/unets.py")
DataModule =loadLib("DataModule", "/mnt/disks/sdb/piCaiCode/model/DataModule.py")
LigtningModel =loadLib("LigtningModel", "/mnt/disks/sdb/piCaiCode/model/LigtningModel.py")
semisuperPreprosess =loadLib("semisuperPreprosess", "/mnt/disks/sdb/piCaiCode/preprocessing/semisuperPreprosess.py")



def getParam(trial,options,key):
    """
    given integer returned from experiment 
    it will look into options dictionary and return required object
    """
    lenn= len(options[key])
    #print(f"  ")
    integerr=trial.suggest_int(key, 0, lenn-1)

    return options[key][integerr]

def addDummyLabelPath(row, labelName, dummyLabelPath):
    """
    adds dummy label to the given column in every spot it is empty
    """
    row = row[1]
    if(row[labelName]==' '):
        return dummyLabelPath
    else:
        return row[labelName]    

def mainTrain(trial,df,experiment_name,dummyDict,num_workers,cpu_num ,default_root_dir,checkpoint_dir,options,num_cpus_per_worker):
    picaiLossArr_auroc_final=[]
    picaiLossArr_AP_final=[]
    picaiLossArr_score_final=[]
    max_epochs=810#100#experiment.get_parameter("max_epochs")
    
    in_channels=4
    out_channels=2




    spacing_keyword="_one_spac_c" 
    sizeWord= "_maxSize_" #config["sizeWord")
    chan3_col_name=f"t2w{spacing_keyword}_3Chan{sizeWord}" 
    chan3_col_name_val=chan3_col_name 
    df=df.loc[df[chan3_col_name] != ' ']
    label_name=f"label{spacing_keyword}{sizeWord}" 
    label_name_val=label_name
    cacheDir =  f"/home/sliceruser/preprocess/monai_persistent_Dataset/{spacing_keyword}/{sizeWord}"
    csvPath = "/mnt/disks/sdb/csvResD.csv"

    # imageRef_path=list(filter(lambda it: it!= '', df[label_name].to_numpy()))[0]
    # dummyLabelPath='/mnt/disks/sdb/dummyData/zeroLabel.nii.gz'
    # sizz=semisuperPreprosess.writeDummyLabels(dummyLabelPath,imageRef_path)
    # img_size = sizz#(sizz[2],sizz[1],sizz[0])
    dummyLabelPath,img_size=dummyDict[spacing_keyword]
    print(f"aaaaaa  img_size {img_size}  {type(img_size)}")

    is_whole_to_train= (sizeWord=="_maxSize_")
    centerCropSize=(81.0, 160.0, 192.0)#=getParam(experiment,options,"centerCropSize",df)
    net= getParam(trial,options,"models") #options["models"][0]#   
    
    # strides=getParam(config,options,"stridesAndChannels",df)["strides"]
    # channels=getParam(config,options,"stridesAndChannels",df)["channels"]
    num_res_units= 0#config["num_res_units")
    act = (Act.PRELU, {"init": 0.2}) #getParam(config,options,"act",df)
    norm= (Norm.BATCH, {}) #getParam(config,options,"norm",df)
    dropout= trial.suggest_float("dropout", 0.0,0.6)
    print(f"aaaaaaaaaaaaaaaaaaa dropout {dropout}")
    to_onehot_y_loss= False

    spacing_keyword="_one_spac_c" 
    sizeWord= "_maxSize_" #config["sizeWord")
    chan3_col_name=f"t2w{spacing_keyword}_3Chan{sizeWord}" 
    chan3_col_name_val=chan3_col_name 
    t2wColName="t2w"+spacing_keyword 
    adcColName="adc"+spacing_keyword
    hbvColName="hbv"+spacing_keyword

    
    df=df.str.replace('/home/sliceruser/data','/mnt/disks/sdb',  regex=True)
    df=df.loc[df[t2wColName] != ' ']
    label_name="label"+spacing_keyword
    label_name_val=label_name
    df=df.loc[df[label_name_val] != ' ']
    df=df.loc[df[t2wColName] != ' ']
    df=df.loc[df[adcColName] != ' ']
    df=df.loc[df[hbvColName] != ' ']
    df=df.loc[df['isAnythingInAnnotated']>0]

    criterion= monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)# Our seg labels are single channel images indicating class index, rather than one-hot
    optimizer_class= torch.optim.NAdam#(lr=config["lr"])#getParam(config,options,"optimizer_class",df)(config["lr"])
    regression_channels=getParam(trial,options,"regression_channels") #options["regression_channels"][1] # getParam(config,options,"regression_channels")#options["regression_channels"][1] #getParam(config,options,"regression_channels",df)
    accumulate_grad_batches=3#config["accumulate_grad_batches"]
    gradient_clip_val= trial.suggest_float("gradient_clip_val",0.0,1.0) #config["gradient_clip_val"]# 0.5,2.0
    net=net(dropout,img_size,in_channels,out_channels)


    RandGaussianNoised_prob= 0.0#trial.suggest_float("RandGaussianNoised_prob", 0.0, 1.0)
    RandAdjustContrastd_prob=trial.suggest_float("RandAdjustContrastd_prob", 0.0, 0.6)
    RandGaussianSmoothd_prob=trial.suggest_float("RandGaussianSmoothd_prob", 0.0, 0.6)
    RandRicianNoised_prob=trial.suggest_float("RandRicianNoised_prob", 0.0, 0.6)
    RandFlipd_prob=trial.suggest_float("RandFlipd_prob", 0.0, 0.6)
    RandAffined_prob=trial.suggest_float("RandAffined_prob", 0.0, 0.6)
    RandCoarseDropoutd_prob= 0.0#trial.suggest_float("RandCoarseDropoutd_prob", 0.0, 1.0)
    RandomElasticDeformation_prob=trial.suggest_float("RandomElasticDeformation_prob", 0.0, 0.6)
    RandomAnisotropy_prob=trial.suggest_float("RandomAnisotropy_prob", 0.0, 0.6)
    RandomMotion_prob=trial.suggest_float("RandomMotion_prob", 0.0, 0.6)
    RandomGhosting_prob=trial.suggest_float("RandomGhosting_prob", 0.0, 0.6)
    RandomSpike_prob=trial.suggest_float("RandomSpike_prob", 0.0, 0.6)
    RandomBiasField_prob=trial.suggest_float("RandomBiasField_prob", 0.0, 0.6)


    os.makedirs('/mnt/disks/sdb/temp', exist_ok = True)
    df[label_name]=list(map(partial(addDummyLabelPath,labelName=label_name ,dummyLabelPath= dummyLabelPath ) ,list(df.iterrows())) )

    ThreeChanNoExperiment.train_model(label_name, dummyLabelPath, df,percentSplit,cacheDir
         ,chan3_col_name,chan3_col_name_val,label_name_val
         ,RandGaussianNoised_prob,RandAdjustContrastd_prob,RandGaussianSmoothd_prob,
         RandRicianNoised_prob,RandFlipd_prob, RandAffined_prob,RandCoarseDropoutd_prob
         ,is_whole_to_train,centerCropSize,
        num_res_units,act,norm,dropout
         ,criterion, optimizer_class,max_epochs,accumulate_grad_batches,gradient_clip_val
         ,picaiLossArr_auroc_final,picaiLossArr_AP_final,picaiLossArr_score_final
          ,experiment_name ,net    ,RandomElasticDeformation_prob
    ,RandomAnisotropy_prob
    ,RandomMotion_prob
    ,RandomGhosting_prob
    ,RandomSpike_prob
    ,RandomBiasField_prob,regression_channels,num_workers,cpu_num ,default_root_dir,checkpoint_dir,trial.suggest_float("lr", 1e-5, 1e-3),num_cpus_per_worker,trial,t2wColName,adcColName,hbvColName)
    # if(len(picaiLossArr_auroc_final)>0):
    #     experiment.log_metric("last_val_loss_auroc",np.nanmax(picaiLossArr_auroc_final))
    #     experiment.log_metric("last_val_loss_Ap",np.nanmax(picaiLossArr_AP_final))
    # experiment.log_metric("last_val_loss_score",np.nanmax(picaiLossArr_score_final))
    


    ###### backup save the optimizer data  #######
    dfOut = pd.DataFrame( columns = ["chan3_col_name","RandGaussianNoised_prob","RandAdjustContrastd_prob",
      "RandGaussianSmoothd_prob","RandRicianNoised_prob", "RandFlipd_prob",
       "RandAffined_prob","RandCoarseDropoutd_prob","dropout", "accumulate_grad_batches"
        "gradient_clip_val","RandomElasticDeformation_prob", 
         "RandomAnisotropy_prob","RandomMotion_prob","RandomGhosting_prob","RandomSpike_prob"
         ,"RandomBiasField_prob","models","criterion","optimizer_class", "regression_channels","last_val_loss_score"       ])
    if(pathOs.exists(csvPath)):


        dfOut=pd.read_csv(csvPath)

    # series= {"chan3_col_name" :chan3_col_name
    #         ,"RandGaussianNoised_prob" :RandGaussianNoised_prob
    #         ,"RandAdjustContrastd_prob" :RandAdjustContrastd_prob
    #         ,"RandGaussianSmoothd_prob":RandGaussianSmoothd_prob
    #         ,"RandRicianNoised_prob" :RandRicianNoised_prob
    #         ,"RandFlipd_prob" :RandFlipd_prob
    #         , "RandAffined_prob" :RandAffined_prob
    #         ,"RandCoarseDropoutd_prob" :RandCoarseDropoutd_prob
    #         ,"dropout" :dropout
    #         ,"accumulate_grad_batches" :accumulate_grad_batches
    #         ,"gradient_clip_val" :gradient_clip_val
    #         ,"RandomElasticDeformation_prob" :RandomElasticDeformation_prob
    #         ,"RandomAnisotropy_prob" :RandomAnisotropy_prob
    #         ,"RandomMotion_prob" :RandomMotion_prob
    #         ,"RandomGhosting_prob" :RandomGhosting_prob
    #         ,"RandomSpike_prob" :RandomSpike_prob
    #         ,"RandomBiasField_prob" :RandomBiasField_prob
    #         #,"models" : config["models"]
    #        # ,"criterion": config["lossF"]
    #        # ,"optimizer_class" : config["optimizer_class"]
    #         #,"regression_channels" :config["regression_channels"]
    #         ,"last_val_loss_score":np.nanmax(picaiLossArr_score_final)   }
    # dfOut=dfOut.append(series, ignore_index = True)
    # dfOut.to_csv(csvPath)


    #experiment.log_parameters(parameters)  
    # experiment.end()
    #removing dummy label 
    #shutil.rmtree('/mnt/disks/sdb/temp') 

    # #evaluating on test dataset
    # with torch.no_grad():   
    # for batch in data.test_dataloader():
    #     inputs = batch['image'][tio.DATA].to(device)
    #     labels = model.net(inputs).argmax(dim=1, keepdim=True).cpu()
    #     for i in range(len(inputs)):
    #         break
    #     break   
    return np.nanmax(picaiLossArr_score_final)

#experiment.end()
