### Define Data Handling

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
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import time

from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from picai_eval import evaluate
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
import tempfile
import shutil
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)
from torchmetrics import Precision
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
import torchio
import importlib.util
import sys
import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
import functools
import operator
from torch.nn.intrinsic.qat import ConvBnReLU3d

import multiprocessing as mp
import time




# def my_task(v):
#     time.sleep(v)
#     return v ** 2


# lenn=8
# squares=[None] * lenn

# TIMEOUT = 2# second timeout
# with mp.Pool(processes = mp.cpu_count()) as pool:
#     results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
    
#     for i in range(lenn):
#         try:
#             return_value = results[i].get(2) # wait for up to time_to_wait seconds
#         except mp.TimeoutError:
#             print('Timeout for v = ', i)
#         else:
#             squares[i]=return_value
#             print(f'Return value for v = {i} is {return_value}')

#     # it = pool.imap(my_task, range(lenn))
#     # squares=list(map(lambda ind :getNext(it,TIMEOUT) ,list(range(lenn)) ))
# print(squares)



import time
from functools import partial
from torchmetrics.functional import precision_recall
from torch.utils.cpp_extension import load
import torchmetrics
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
from monai.metrics import (
    ConfusionMatrixMetric,
    compute_confusion_matrix_metric,
    do_metric_reduction,
    get_confusion_matrix,
)

import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

from picai_eval.eval import evaluate_case


class UNetToRegresion(nn.Module):
    def __init__(self,
        in_channels,
        regression_channels
        ,segmModel
    ) -> None:
        super().__init__()
        self.segmModel=segmModel
        self.model = nn.Sequential(
            ConvBnReLU3d(in_channels=in_channels, out_channels=regression_channels[0], kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[0], out_channels=regression_channels[1], kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[1], out_channels=regression_channels[2], kernel_size=3, stride=1,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[2], out_channels=1, kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            nn.AdaptiveMaxPool3d((8,8,2)),#ensuring such dimension 
            nn.Flatten(),
            #nn.BatchNorm3d(8*8*4),
            nn.Linear(in_features=8*8*2, out_features=100),
            #nn.BatchNorm3d(100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=1)
        )
    def forward(self, x):
        segmMap=self.segmModel(x)
        #print(f"segmMap  {segmMap}")
        return (segmMap,self.model(segmMap))


# torch.autograd.set_detect_anomaly(True)

def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def saveFilesInDir(gold_arr,y_hat_arr, directory, patId,imageArr, hatPostA):
    """
    saves arrays in given directory and return paths to them
    """
    # gold_im_path = join(directory, patId+ "_gold.npy" )
    # yHat_im_path = join(directory, patId+ "_hat.npy" )
    # np.save(gold_im_path, gold_arr)
    # np.save(yHat_im_path, y_hat_arr)
    gold_im_path = join(directory, patId+ "_gold.nii.gz" )
    yHat_im_path =join(directory, patId+ "_hat.nii.gz" )
    image_path =join(directory, patId+ "image.nii.gz" )
    hatPostA_path =join(directory, patId+ "hatPostA.nii.gz" )
    print(f"suum hat  {np.sum( y_hat_arr.numpy())} hatPostA {np.sum(hatPostA)}  ")
    # gold_arr=np.swapaxes(gold_arr,0,2)
    # y_hat_arr=np.swapaxes(y_hat_arr,0,2)
    # print(f"uniq gold { gold_arr.shape  }   yhat { y_hat_arr.shape }   yhat maxes  {np.maximum(y_hat_arr)}  hyat min {np.minimum(y_hat_arr)} ")
    gold_arr=gold_arr[1,:,:,:].numpy()
    y_hat_arr=y_hat_arr[1,:,:,:].numpy()

    gold_arr=np.swapaxes(gold_arr,0,2)
    y_hat_arr=np.swapaxes(y_hat_arr,0,2)
    
    image = sitk.GetImageFromArray(gold_arr)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(gold_im_path)
    writer.Execute(image)


    image = sitk.GetImageFromArray(y_hat_arr)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(yHat_im_path)
    writer.Execute(image) 

    image = sitk.GetImageFromArray(np.swapaxes(imageArr[0,:,:,:].numpy(),0,2))
    writer = sitk.ImageFileWriter()
    writer.SetFileName(image_path)
    writer.Execute(image)

    image = sitk.GetImageFromArray(np.swapaxes(hatPostA,0,2))
    writer = sitk.ImageFileWriter()
    writer.SetFileName(hatPostA_path)
    writer.Execute(image)

    return(gold_im_path,yHat_im_path)


# def saveToValidate(i,y_det,regress_res_cpu,temp_val_dir,y_true,patIds):
#     y_det_curr=y_det[i]
#     #TODO unhash
#     if(np.rint(regress_res_cpu[i])==0):
#         y_det_curr=np.zeros_like(y_det_curr)
#     return saveFilesInDir(y_true[i],y_det_curr, temp_val_dir, patIds[i])

def getArrayFromPath(path):
    image1=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image1)

def extractLesions_my(x):
    return extract_lesion_candidates(x)[0]

def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,images,hatPostA):
# def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,reg_hat):
    return saveFilesInDir(y_true[i],y_det[i], temp_val_dir, patIds[i],images[i],hatPostA[i])
    
    # if(reg_hat[i]>0):
    #     return saveFilesInDir(y_true[i],y_det[i], temp_val_dir, patIds[i])
    # #when it is equal 0 we zero out the result
    # return saveFilesInDir(y_true[i],np.zeros_like(y_det[i]), temp_val_dir, patIds[i])    

# def calcDiceFromPaths(i,list_yHat_val,list_gold_val):

#     y_hat= monai.transforms.LoadImage()(list_yHat_val[i])
#     gold_val= monai.transforms.LoadImage()(list_gold_val[i])

#     print(f" yHat_val {y_hat[0]} gold_val {gold_val[0]} ")
    
#     postProcessHat=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])
#     load_true=monai.transforms.Compose([AsDiscrete( to_onehot=2)])
#     return monai.metrics.compute_generalized_dice( postProcessHat(y_hat) ,load_true(gold_val))

def evaluate_case_for_map(i,y_det,y_true):
    print("evaluate_case_for_map") 
    return evaluate_case(y_det=y_det[i] 
                        ,y_true=y_true[i] 
                        ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

def getNext(i,results,TIMEOUT):
    try:
        # return it.next(timeout=TIMEOUT)
        return results[i].get(TIMEOUT)

    except:
        print("timed outt ")
        return None    


def processDice(i,postProcess,y_det,y_true):
    try:
        hatPost=postProcess(y_det[i])
        # print( f" hatPost {hatPost.size()}  y_true {y_true[i].cpu().size()} " )
        locDice=monai.metrics.compute_generalized_dice( hatPost ,y_true[i])
        return (locDice,hatPost.numpy())
    except:
        return (0.0,np.zeros_like(y_det[i]))
    # avSurface_dist_loc=monai.metrics.compute_average_surface_distance(hatPost, y_true[i])
    #monai.metrics.compute_generalized_dice(
    # self.rocAuc(hatPost.cpu() ,y_true[i].cpu())
    # self.dices.append(locDice)
    # # self.surfDists.append(avSurface_dist_loc)
    # hatPostA.append(hatPost[1,:,:,:])    

class Model(pl.LightningModule):
    def __init__(self
    , net
    , criterion
    , learning_rate
    , optimizer_class
    ,picaiLossArr_auroc_final
    ,picaiLossArr_AP_final
    ,picaiLossArr_score_final
    ):
        super().__init__()
        self.lr = learning_rate
        self.net=net
        # self.modelRegression = UNetToRegresion(2,regression_channels,net)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = monai.metrics.GeneralizedDiceScore()
        #self.rocAuc=monai.metrics.ROCAUCMetric()
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]
        self.dices=[]
        self.surfDists=[]
        
        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final
        #temporary directory for validation images and their labels
        self.temp_val_dir= '/mnt/disks/sdb/tempE' #tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.list_back_yHat_val=[]
        self.isAnyNan=False
        #os.makedirs('/mnt/disks/sdb/temp')
        # self.postProcess=monai.transforms.Compose([EnsureType(), monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        # self.postProcess=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postProcess=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postTrue = Compose([EnsureType()])
        #self.F1Score = torchmetrics.F1Score()

        os.makedirs(self.temp_val_dir,  exist_ok = True)             
        shutil.rmtree(self.temp_val_dir) 
        os.makedirs(self.temp_val_dir,  exist_ok = True)             

    def configure_optimizers(self):
        # optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        optimizer = self.optimizer_class(self.parameters())
        return optimizer
    

    
    # def infer_batch_pos(self, batch):
    #     x, y, numLesions = batch["pos"]['chan3_col_name'], batch["pos"]['label'], batch["pos"]['num_lesions_to_retain']
    #     y_hat = self.net(x)
    #     return y_hat, y, numLesions



    # def infer_batch_all(self, batch):
    #     x, numLesions =batch["all"]['chan3_col_name'], batch["all"]['num_lesions_to_retain']
    #     y_hat = self.net(x)
    #     return y_hat, numLesions

    def calcLossHelp(self,isAnythingInAnnotated_list,seg_hat_list, y_true_list,i):
        if(isAnythingInAnnotated_list[i]>0):
            return self.criterion(seg_hat_list[i], y_true_list[i])
        return ' '    
        #     lossReg=F.smooth_l1_loss(torch.Tensor(reg_hat_list[i]).int().to(self.device) , torch.Tensor(int(numLesions_list[i])).int().to(self.device) ) 
        #     return torch.add(lossSeg,lossReg)
        # return  F.smooth_l1_loss(torch.Tensor(reg_hat_list[i]).int().to(self.device) , torch.Tensor(int(numLesions_list[i])).int().to(self.device) ) 



    def calculateLoss(self,isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions):
        return self.criterion(seg_hat,y_true)+ F.smooth_l1_loss(reg_hat.flatten(),torch.Tensor(numLesions).to(self.device).flatten() )

        # seg_hat_list = decollate_batch(seg_hat)
        # isAnythingInAnnotated_list = decollate_batch(isAnythingInAnnotated)
        # y_true_list = decollate_batch(y_true)
        # toSum= list(map(lambda i:  self.calcLossHelp(isAnythingInAnnotated_list,seg_hat_list, y_true_list ,i) , list( range(0,len( seg_hat_list)) )))
        # toSum= list(filter(lambda it: it!=' '  ,toSum))
        # if(len(toSum)>0):
        #     segLoss= torch.sum(torch.stack(toSum))
        #     lossReg=F.smooth_l1_loss(reg_hat.flatten(),torch.Tensor(numLesions).to(self.device).flatten())*2
        #     return torch.add(segLoss,lossReg)

        # #print(f"reg_hat {reg_hat} numLesions{numLesions}  "  )        
        # return F.smooth_l1_loss(reg_hat.flatten(),torch.Tensor(numLesions).to(self.device).flatten() )*2


    def training_step(self, batch, batch_idx):
        # every second iteration we will do the training for segmentation
        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name'], batch['label'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        # seg_hat, reg_hat = self.modelRegression(x)
        seg_hat = self.net(x)
        return self.criterion(seg_hat,y_true)
        #return self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)

        # if(isAnythingInAnnotated>0):
        #     lossSeg=self.criterion(seg_hat, y_true)
        #     lossReg=F.smooth_l1_loss(reg_hat,numLesions)
        #     return torch.add(lossSeg,lossReg)
        # return  F.smooth_l1_loss(reg_hat,numLesions)   
        
        # # y_hat, y , numLesions_ab= self.infer_batch_pos(batch)
        # lossa = self.criterion(y_hat, y)
        # # regressab=  self.modelRegression(y_hat)
        # # numLesions_ab2=list(map(lambda entry : int(entry), numLesions_ab ))
        # # numLesions_ab3=torch.Tensor(numLesions_ab2).to(self.device)  
        # # lossab=F.smooth_l1_loss(torch.flatten(regressab), torch.flatten(numLesions_ab3) )
      
        # # # in case we have odd iteration we get access only to number of lesions present in the image not where they are (if they are present at all)    
        # # y_hat_all, numLesions= self.infer_batch_all(batch)
        # # regress_res=self.modelRegression(y_hat_all)
        # # numLesions1=list(map(lambda entry : int(entry), numLesions ))
        # # numLesions2=torch.Tensor(numLesions1).to(self.device)
        # # # print(f" regress res {torch.flatten(regress_res).size()}  orig {torch.flatten(numLesions).size() } ")
        # # lossb=F.smooth_l1_loss(torch.flatten(regress_res), torch.flatten(numLesions2) )

        # # self.log('train_loss', torch.add(lossa,lossb), prog_bar=True)
        # # self.log('train_image_loss', lossa, prog_bar=True)
        # # self.log('train_reg_loss', lossb, prog_bar=True)
        # # return torch.add(torch.add(lossa,lossb),lossab)
        # return lossa
    # def validation_step(self, batch, batch_idx):
        # self.list_gold_val.append(tupl[0])
        # self.list_yHat_val.append(tupl[1])


    def validation_step(self, batch, batch_idx):
        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name_val'], batch['label_name_val'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        
        #seg_hat, reg_hat = self.modelRegression(x)        
        # seg_hat, reg_hat = self.modelRegression(x)        
        seg_hat = self.net(x).cpu().detach()
        seg_hat=torch.sigmoid(seg_hat).cpu().detach()

        #loss= self.criterion(seg_hat,y_true)# self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)      

        # locDice = np.mean(monai.metrics.compute_generalized_dice( self.postProcess(seg_hat) ,  y_true  ).numpy())
        # print(f"locDice {locDice}")
        
        

#         #we want only first channel
#         y_true=y_true[:,1,:,:,:].cpu().detach()
#         y_det=seg_hat[:,1,:,:,:].cpu().detach()

        y_det = decollate_batch(seg_hat.cpu().detach())
        # y_background = decollate_batch(seg_hat[:,0,:,:,:].cpu().detach())
        y_true = decollate_batch(y_true.cpu().detach())
        patIds = decollate_batch(batch['patient_id'])

        images = decollate_batch(x.cpu().detach()) 
#         # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        # hatPostA=[]
        # for i in range(0,len(y_det)):
        #     hatPost=self.postProcess(y_det[i])
        #     # print( f" hatPost {hatPost.size()}  y_true {y_true[i].cpu().size()} " )
        #     locDice=monai.metrics.compute_generalized_dice( hatPost ,y_true[i])
        #     # avSurface_dist_loc=monai.metrics.compute_average_surface_distance(hatPost, y_true[i])
        #     #monai.metrics.compute_generalized_dice(
        #     # self.rocAuc(hatPost.cpu() ,y_true[i].cpu())
        #     self.dices.append(locDice)
        #     # self.surfDists.append(avSurface_dist_loc)
        #     hatPostA.append(hatPost[1,:,:,:])


        #     self.dices.append(locDice)


        pathssList=[]
        dicesList=[]
        hatPostA=[]
        # with mp.Pool(processes = mp.cpu_count()) as pool:
        #     # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
        #     dicesList=pool.map(partial(processDice,postProcess=self.postProcess,y_det=y_det, y_true=y_true ),list(range(0,len(y_true))))
        dicesList=list(map(partial(processDice,postProcess=self.postProcess,y_det=y_det, y_true=y_true ),list(range(0,len(y_true)))))

        hatPostA=list(map(lambda tupl: tupl[1],dicesList ))
        dicees=list(map(lambda tupl: tupl[0],dicesList ))
        
        with mp.Pool(processes = mp.cpu_count()) as pool:        
            pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,images=images,hatPostA=hatPostA),list(range(0,len(y_true))))

        forGoldVal=list(map(lambda tupl :tupl[0] ,pathssList  ))
        fory_hatVal=list(map(lambda tupl :tupl[1] ,pathssList  ))
        # fory__bach_hatVal=list(map(lambda tupl :tupl[2] ,pathssList  ))

        



        


# #         # self.list_gold_val=self.list_gold_val+forGoldVal
# #         # self.list_yHat_val=self.list_gold_val+fory_hatVal

# # # save_candidates_to_dir(y_true,y_det,patIds,i,temp_val_dir)
        for i in range(0,len(y_true)):
            # tupl=saveFilesInDir(y_true[i],y_det[i], self.temp_val_dir, patIds[i])
            # print("saving entry   ")
            # self.list_gold_val.append(tupl[0])
            # self.list_yHat_val.append(tupl[1])
            self.list_gold_val.append(forGoldVal[i])
            self.list_yHat_val.append(fory_hatVal[i])
            self.dices.append(dicees[i])
            # self.list_back_yHat_val.append(fory__bach_hatVal[i])
# #         self.log('val_loss', loss )

#        # return {'loss' :loss,'loc_dice': diceVall }

        #TODO probably this [1,:,:,:] could break the evaluation ...
        # y_det=[x.cpu().detach().numpy()[1,:,:,:][0] for x in y_det]
        # y_true=[x.cpu().detach().numpy() for x in y_true]
        # y_det= list(map(self.postProcess  , y_det))
        # y_true= list(map(self.postTrue , y_det))


        # if(torch.sum(torch.isnan( y_det))>0):
        #     self.isAnyNan=True

        # regress_res2= torch.flatten(reg_hat) 
        # regress_res3=list(map(lambda el:round(el) ,torch.flatten(regress_res2).cpu().detach().numpy() ))

        # total_loss=precision_recall(torch.Tensor(regress_res3).int(), torch.Tensor(numLesions).cpu().int(), average='macro', num_classes=4)
        # total_loss1=torch.mean(torch.stack([total_loss[0],total_loss[1]] ))#self.F1Score
        
        # if(torch.sum(isAnythingInAnnotated)>0):
        #     dice = DiceMetric()
        #     for i in range(0,len( y_det)):
        #         if(isAnythingInAnnotated[i]>0):
        #             y_det_i=self.postProcess(y_det[i])[0,:,:,:].cpu()
        #             y_true_i=self.postTrue(y_true[i])[1,:,:,:].cpu()
        #             if(torch.sum(y_det_i).item()>0 and torch.sum(y_true_i).item()>0 ):
        #                 dice(y_det_i,y_true_i)

        #     self.log("dice", dice.aggregate())
        #     #print(f" total loss a {total_loss1} val_loss {val_losss}  dice.aggregate() {dice.aggregate()}")
        #     total_loss2= torch.add(total_loss1,dice.aggregate())
        #     print(f" total loss b {total_loss2}  total_loss,dice.aggregate() {dice.aggregate()}")
            
        #     self.picaiLossArr_score_final.append(total_loss2.item())
        #     return {'val_acc': total_loss2.item(), 'val_loss':val_losss}
        
        # #in case no positive segmentation information is available
        # self.picaiLossArr_score_final.append(total_loss1.item())
        # return {'val_acc': total_loss1.item(), 'val_loss':val_losss}

    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")

        #self.log('dice', np.mean(self.dices))
        # self.dice_metric.reset()

 
        # print( f"rocAuc  {self.rocAuc.aggregate().item()}"  )
        # #self.log('precision ', monai.metrics.compute_confusion_matrix_metric("precision", confusion_matrix) )
        # self.rocAuc.reset()        


        
        #print(f" self.list_yHat_val {self.list_yHat_val} ")
        if(len(self.list_yHat_val)>1 and (not self.isAnyNan)):
        # if(False):
            # with mp.Pool(processes = mp.cpu_count()) as pool:
            #     dices=pool.map(partial(calcDiceFromPaths,list_yHat_val=self.list_yHat_val,list_gold_val=self.list_gold_val   ),list(range(0,len(self.list_yHat_val))))
            # dices=list(map(partial(calcDiceFromPaths,list_yHat_val=self.list_yHat_val,list_gold_val=self.list_gold_val   ),list(range(0,len(self.list_yHat_val)))))
            # meanDice=torch.mean(torch.stack( dices)).item()
            
            self.log('meanDice',torch.mean(torch.stack( dices)).item() )
            print('meanDice',np.mean( np.array(self.dices ).flatten()))
            # self.log('mean_surface_distance',torch.mean(torch.stack( self.surfDists)).item())

            lenn=len(self.list_yHat_val)
            numPerIter=1
            numIters=math.ceil(lenn/numPerIter)-1



            meanPiecaiMetr_auroc_list=[]
            meanPiecaiMetr_AP_list=[]
            meanPiecaiMetr_score_list=[]
            print(f" numIters {numIters} ")
            
            pool = mp.Pool()
            listPerEval=[None] * lenn

            # #timeout based on https://stackoverflow.com/questions/66051638/set-a-time-limit-on-the-pool-map-operation-when-using-multiprocessing
            my_task=partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val)
            # def my_callback(t):
            #     print(f"tttttt  {t}")
            #     s, i = t
            #     listPerEval[i] = s
            # results=[pool.apply_async(my_task, args=(i,), callback=my_callback) for i in list(range(0,lenn))]
            # TIMEOUT = 300# second timeout
            # time.sleep(TIMEOUT)
            # pool.terminate()
            # #filtering out those that timed out
            # listPerEval=list(filter(lambda it:it!=None,listPerEval))
            # print(f" results timed out {lenn-len(listPerEval)} from all {lenn} ")

            TIMEOUT = 50# second timeout


# TIMEOUT = 2# second timeout
# with mp.Pool(processes = mp.cpu_count()) as pool:
#     results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
    
#     for i in range(lenn):
#         try:
#             return_value = results[i].get(2) # wait for up to time_to_wait seconds
#         except mp.TimeoutError:
#             print('Timeout for v = ', i)
#         else:
#             squares[i]=return_value
#             print(f'Return value for v = {i} is {return_value}')


#     # it = pool.imap(my_task, range(lenn))
#     # squares=list(map(lambda ind :getNext(it,TIMEOUT) ,list(range(lenn)) ))
# print(squares)


            with mp.Pool(processes = mp.cpu_count()) as pool:
                #it = pool.imap(my_task, range(lenn))
                results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
                time.sleep(TIMEOUT)
                listPerEval=list(map(lambda ind :getNext(ind,results,5) ,list(range(lenn)) ))
            #filtering out those that timed out
            listPerEval=list(filter(lambda it:it!=None,listPerEval))
            print(f" results timed out {lenn-len(listPerEval)} from all {lenn} ")                
                    # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
                # listPerEval=pool.map( partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val) , list(range(0,lenn)))


            # listPerEval=list(map( partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val) , list(range(0,lenn))))


            # initialize placeholders
            case_target: Dict[Hashable, int] = {}
            case_weight: Dict[Hashable, float] = {}
            case_pred: Dict[Hashable, float] = {}
            lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
            lesion_weight: Dict[Hashable, List[float]] = {}

            meanPiecaiMetr_auroc=0.0
            meanPiecaiMetr_AP=0.0
            meanPiecaiMetr_score=0.0

            idx=0
            if(len(listPerEval)>0):
                for pairr in listPerEval:
                    idx+=1
                    lesion_results_case, case_confidence = pairr

                    case_weight[idx] = 1.0
                    case_pred[idx] = case_confidence
                    if len(lesion_results_case):
                        case_target[idx] = np.max([a[0] for a in lesion_results_case])
                    else:
                        case_target[idx] = 0

                    # accumulate outputs
                    lesion_results[idx] = lesion_results_case
                    lesion_weight[idx] = [1.0] * len(lesion_results_case)

                # collect results in a Metrics object
                valid_metrics = Metrics(
                    lesion_results=lesion_results,
                    case_target=case_target,
                    case_pred=case_pred,
                    case_weight=case_weight,
                    lesion_weight=lesion_weight
                )




                # for i in range(0,numIters):
                #     valid_metrics = evaluate(y_det=self.list_yHat_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
                #                         y_true=self.list_gold_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
                #                         num_parallel_calls= min(numPerIter,os.cpu_count())
                #                         ,verbose=1
                #                         #,y_true_postprocess_func=lambda pred: pred[1,:,:,:]
                #                         #y_true=iter(y_true),
                #                         ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
                #                         #,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
                #                         )
                # meanPiecaiMetr_auroc_list.append(valid_metrics.auroc)
                # meanPiecaiMetr_AP_list.append(valid_metrics.AP)
                # meanPiecaiMetr_score_list.append((-1)*valid_metrics.score)
                #print("finished evaluating")

                meanPiecaiMetr_auroc=valid_metrics.auroc
                meanPiecaiMetr_AP=valid_metrics.AP
                meanPiecaiMetr_score=(-1)*valid_metrics.score
            # meanPiecaiMetr_auroc=np.nanmean(meanPiecaiMetr_auroc_list)
            # meanPiecaiMetr_AP=np.nanmean(meanPiecaiMetr_AP_list)
            # meanPiecaiMetr_score=np.nanmean(meanPiecaiMetr_score_list)
        

      
            print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

            self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            self.log('val_mean_AP', meanPiecaiMetr_AP)
            self.log('mean_val_acc', meanPiecaiMetr_score)
            # tensorss = [torch.as_tensor(x['loc_dice']) for x in outputs]
            # if( len(tensorss)>0):
            #     avg_dice = torch.mean(torch.stack(tensorss))

            self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
            self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
            self.picaiLossArr_score_final.append(meanPiecaiMetr_score)

            #resetting to 0 
            self.picaiLossArr_auroc=[]
            self.picaiLossArr_AP=[]
            self.picaiLossArr_score=[]







        #clearing and recreatin temporary directory
        #shutil.rmtree(self.temp_val_dir)   
        #self.temp_val_dir=tempfile.mkdtemp() 
        self.temp_val_dir=pathOs.join('/mnt/disks/sdb/tempE',str(self.trainer.current_epoch))
        os.makedirs(self.temp_val_dir,  exist_ok = True)  


        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.list_back_yHat_val=[]

        #in case we have Nan values training is unstable and we want to terminate it     
        # if(self.isAnyNan):
        #     self.log('val_mean_score', -0.2)
        #     self.picaiLossArr_score_final=[-0.2]
        #     self.picaiLossArr_AP_final=[-0.2]
        #     self.picaiLossArr_auroc_final=[-0.2]
        #     print(" naans in outputt  ")

        #self.isAnyNan=False
        #return {"mean_val_acc": self.log}


        # # avg_loss = torch.mean(torch.stack([torch.as_tensor(x['val_loss']) for x in outputs]))
        # # print(f"mean_val_loss { avg_loss}")
        # # avg_acc = torch.mean(torch.stack([torch.as_tensor(x['val_acc']) for x in outputs]))
        # #val_accs=list(map(lambda x : x['val_acc'],outputs))
        # val_accs=list(map(lambda x : x['val_acc'].cpu().detach().numpy(),outputs))
        # #print(f" a  val_accs {val_accs} ")
        # val_accs=np.nanmean(np.array( val_accs).flatten())
        # #print(f" b  val_accs {val_accs} mean {np.mean(val_accs)}")

        # #avg_acc = np.mean(np.array(([x['val_acc'].cpu().detach().numpy() for x in outputs])).flatten() )

        # # self.log("mean_val_loss", avg_loss)
        # self.log("mean_val_acc", np.mean(val_accs))

        # # self.log('ptl/val_loss', avg_loss)
        # # self.log('ptl/val_accuracy', avg_acc)
        # #return {'mean_val_loss': avg_loss, 'mean_val_acc':avg_acc}

#self.postProcess

#             image1=sitk.ReadImage(path)
# #     data = sitk.GetArrayFromImage(image1)