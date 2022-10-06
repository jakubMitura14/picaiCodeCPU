
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from comet_ml import Optimizer

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
import torch_optimizer as optim
monai.utils.set_determinism()
import geomloss
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

import importlib.util
import sys
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

import preprocessing.ManageMetadata as manageMetaData
import model.unets as unets
import model.DataModule as DataModule
import model.LigtningModel as LigtningModel
import Three_chan_baseline
import preprocessing.semisuperPreprosess as semisuperPreprosess

# manageMetaData =loadLib("ManageMetadata", "/mnt/disks/sdb/piCaiCode/preprocessing/ManageMetadata.py")
# dataUtils =loadLib("dataUtils", "/mnt/disks/sdb/piCaiCode/dataManag/utils/dataUtils.py")
# unets =loadLib("unets", "/mnt/disks/sdb/piCaiCode/model/unets.py")
# DataModule =loadLib("DataModule", "/mnt/disks/sdb/piCaiCode/model/DataModule.py")
# LigtningModel =loadLib("LigtningModel", "/mnt/disks/sdb/piCaiCode/model/LigtningModel.py")
# Three_chan_baseline =loadLib("Three_chan_baseline", "/mnt/disks/sdb/piCaiCode/Three_chan_baseline.py")
# detectSemiSupervised =loadLib("detectSemiSupervised", "/mnt/disks/sdb/piCaiCode/model/detectSemiSupervised.py")
# semisuperPreprosess =loadLib("semisuperPreprosess", "/mnt/disks/sdb/piCaiCode/preprocessing/semisuperPreprosess.py")

# ray.init(runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}})
##options




def getUnetA(dropout,input_image_size,in_channels,out_channels ):
    return unets.UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
        channels=[32, 64, 128, 256, 512, 1024],
        num_res_units= 0,
        act = (Act.PRELU, {"init": 0.2}),
        norm= (Norm.BATCH, {}),
        dropout= dropout
    )
def getUnetB(dropout,input_image_size,in_channels,out_channels):
    return unets.UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
        channels=[32, 64, 128, 256, 512],
        num_res_units= 0,
        act = (Act.PRELU, {"init": 0.2}),
        norm= (Norm.BATCH, {}),
        dropout= dropout
    )
def getAhnet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.AHNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        psp_block_num=3   )

def getSegResNet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
    )
def getSegResNetVAE(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.SegResNetVAE(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        input_image_size=torch.Tensor(input_image_size)

    )


def getAttentionUnet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        # img_size=input_image_size,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
        channels=[32, 64, 128, 256, 512],
        dropout=dropout
    )
def getSwinUNETR(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=input_image_size
    )
def getVNet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.VNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout
    )
def getViTAutoEnc(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.ViTAutoEnc(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=input_image_size,
        patch_size=(16,16,16)
    )


#batch sizes
#unet one spac sth like 16


def getOptNAdam(lr):
    return torch.optim.NAdam(lr=lr)

#getViTAutoEnc,getAhnet,getSegResNetVAE,getAttentionUnet,getSwinUNETR,getSegResNet,getVNet,getUnetB
options={

# "models":[getUnetA,getUnetB,getVNet,getSegResNet],
"models":[getUnetA],# getVNet,getSegResNet,getSwinUNETR
"regression_channels":[[10,16,32]], #[1,1,1],[2,4,8],

# "lossF":[monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
#         # ,SamplesLoss(loss="sinkhorn",p=3)
#         # ,SamplesLoss(loss="hausdorff",p=3)
#         # ,SamplesLoss(loss="energy",p=3)
        
# ],

"optimizer_class": [getOptNAdam] ,#torch.optim.LBFGS ,torch.optim.LBFGS optim.AggMo,   look in https://pytorch-optimizer.readthedocs.io/en/latest/api.html
"act":[(Act.PRELU, {"init": 0.2})],#,(Act.LEAKYRELU, {})                                         
"norm":[(Norm.BATCH, {}) ],
"centerCropSize":[(256, 256,32)],
#TODO() learning rate schedulers https://medium.com/mlearning-ai/make-powerful-deep-learning-models-quickly-using-pytorch-lightning-29f040158ef3
}






df = pd.read_csv("/mnt/disks/sdb/metadata/processedMetaData_current_b.csv")
spacings =  ["_one_spac_c" ,"_med_spac_b" ]# ,"_med_spac_b" #config['parameters']['spacing_keyword']["values"]

def getDummy(spac):
    label_name=f"label{spac}_maxSize_" 
    imageRef_path=list(filter(lambda it: it!= '', df[label_name].to_numpy()))[0].replace('/home/sliceruser/data','/mnt/disks/sdb')
    dummyLabelPath=f"/mnt/disks/sdb/dummyData/zeroLabel{spac}.nii.gz"
    sizz=semisuperPreprosess.writeDummyLabels(dummyLabelPath,imageRef_path)
    img_size = sizz#(sizz[2],sizz[1],sizz[0])
    return(dummyLabelPath,img_size)



aa=list(map(getDummy  ,spacings  ))
dummyDict={"_one_spac_c" :aa[0],"_med_spac_b":aa[1]   }




experiment_name="picai_hp_35"
# Three_chan_baseline.mainTrain(options,df,experiment_name,dummyDict)
num_workers=2
cpu_num=11 #per gpu
default_root_dir='/mnt/disks/sdb/lightninghk'
checkpoint_dir='/mnt/disks/sdb/tuneCheckpoints11'
mainTuneDir='/mnt/disks/sdb/mainTuneDir'
os.makedirs(checkpoint_dir,  exist_ok = True) 
# os.makedirs(default_root_dir,  exist_ok = True) 
num_cpus_per_worker=cpu_num



def objective(trial: optuna.trial.Trial) -> float:

    return Three_chan_baseline.mainTrain(trial,df,experiment_name,dummyDict
    ,num_workers,cpu_num ,default_root_dir,checkpoint_dir,options,num_cpus_per_worker)



study = optuna.create_study(
        study_name=experiment_name
        ,sampler=optuna.samplers.NSGAIISampler()    
        ,pruner=optuna.pruners.HyperbandPruner()
        ,storage="mysql://root@34.147.7.30/picai_hp_36"
        ,load_if_exists=True
        #,storage="mysql://root@127.0.0.1:3306/picai_hp_35"
        )
        #mysql://root@localhost/example
        
study.optimize(objective, n_trials=40)


print("***********  study.best_trial *********")
print(f"study.best_trial {study.trials_dataframe() }")










# def train_mnist_tune(config, num_epochs=10, num_workerss=0, data_dir="~/data"):
#     data_dir = os.path.expanduser(data_dir)
#     model = LightningMNISTClassifier(config, data_dir)
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         # If fractional GPUs passed in, convert to int.
#         gpus=math.ceil(num_workerss),
#         logger=TensorBoardLogger(
#             save_dir=os.getcwd(), name="", version="."),
#         enable_progress_bar=False,
#         callbacks=[
#             TuneReportCallback(
#                 {
#                     "loss": "ptl/val_loss",
#                     "mean_accuracy": "ptl/val_accuracy"
#                 },
#                 on="validation_end")
#         ])
#     trainer.fit(model)
