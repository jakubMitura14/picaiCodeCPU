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
import functools
import multiprocessing as mp
import os
import os.path
monai.utils.set_determinism()
from functools import partial
from pytorch_lightning.loggers import CometLogger
from optuna.integration import PyTorchLightningPruningCallback

# import preprocessing.transformsForMain
# import preprocessing.ManageMetadata
import model.unets as unets
import model.DataModule as DataModule
import model.LigtningModel as LigtningModel
# import preprocessing.semisuperPreprosess


def isAnnytingInAnnotatedInner(row,colName):
    row=row[1]
    path=row[colName]
    image1 = sitk.ReadImage(path)
    #image1 = sitk.Cast(image1, sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(image1)
    return np.sum(data)




def train_model(label_name, dummyLabelPath, df,percentSplit,cacheDir
         ,chan3_col_name,chan3_col_name_val,label_name_val
         ,RandGaussianNoised_prob,RandAdjustContrastd_prob,RandGaussianSmoothd_prob,
         RandRicianNoised_prob,RandFlipd_prob, RandAffined_prob,RandCoarseDropoutd_prob
         ,is_whole_to_train,centerCropSize,
         num_res_units,act,norm,dropout
         ,criterion, optimizer_class,max_epochs,accumulate_grad_batches,gradient_clip_val
         ,picaiLossArr_auroc_final,picaiLossArr_AP_final,picaiLossArr_score_final
          ,experiment_name,net    ,RandomElasticDeformation_prob
    ,RandomAnisotropy_prob
    ,RandomMotion_prob
    ,RandomGhosting_prob
    ,RandomSpike_prob
    ,RandomBiasField_prob,regression_channels,num_workers,cpu_num ,default_root_dir,checkpoint_dir,lr,num_cpus_per_worker,trial,t2wColName,adcColName,hbvColName):        
    #TODO(remove)
    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=experiment_name, # Optional
        #experiment_name="baseline" # Optional
    )
    
    data = DataModule.PiCaiDataModule(
        df= df,
        batch_size=12,#
        trainSizePercent=percentSplit,# 
        num_workers=os.cpu_count(),#os.cpu_count(),
        drop_last=False,#True,
        #we need to use diffrent cache folders depending on weather we are dividing data or not
        cache_dir=cacheDir,
        chan3_col_name =chan3_col_name,
        chan3_col_name_val=chan3_col_name_val,
        label_name_val=label_name_val,
        label_name=label_name
        ,t2wColName=t2wColName
        ,adcColName=adcColName
        ,hbvColName=hbvColName       
        #maxSize=maxSize
        ,RandGaussianNoised_prob=RandGaussianNoised_prob
        ,RandAdjustContrastd_prob=RandAdjustContrastd_prob
        ,RandGaussianSmoothd_prob=RandGaussianSmoothd_prob
        ,RandRicianNoised_prob=RandRicianNoised_prob
        ,RandFlipd_prob=RandFlipd_prob
        ,RandAffined_prob=RandAffined_prob
        ,RandCoarseDropoutd_prob=RandCoarseDropoutd_prob
        ,is_whole_to_train=is_whole_to_train
        ,centerCropSize=centerCropSize
        ,RandomElasticDeformation_prob=RandomElasticDeformation_prob
        ,RandomAnisotropy_prob=RandomAnisotropy_prob
        ,RandomMotion_prob=RandomMotion_prob
        ,RandomGhosting_prob=RandomGhosting_prob
        ,RandomSpike_prob=RandomSpike_prob
        ,RandomBiasField_prob=RandomBiasField_prob
    )


    data.prepare_data()
    data.setup()
    # definition described in model folder
    # from https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/unet/training_setup/neural_networks/unets.py
    # unet= unets.UNet(
    #     spatial_dims=3,
    #     in_channels=3,
    #     out_channels=2,
    #     strides=strides,
    #     channels=channels,
    #     num_res_units= num_res_units,
    #     act = act,
    #     norm= norm,
    #     dropout= dropout
    # )

    model = LigtningModel.Model(
        net=net,
        criterion=  criterion,# Our seg labels are single channel images indicating class index, rather than one-hot
        learning_rate=1e-2,
        optimizer_class= optimizer_class,
        picaiLossArr_auroc_final=picaiLossArr_auroc_final,
        picaiLossArr_AP_final=picaiLossArr_AP_final,
        picaiLossArr_score_final=picaiLossArr_score_final,
        regression_channels=regression_channels,
        lr=lr,
        trial=trial
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='mean_val_acc',
        patience=7,
        mode="max",
        #divergence_threshold=(-0.1)
    )


    # tuneCallBack=TuneReportCheckpointCallback(
    #     metrics={
    #         "mean_val_loss": "mean_val_loss",
    #         "mean_val_acc": "mean_val_acc"
    #     },
    #     filename="checkpointtt.ckpt",
    #     on="validation_end")
    # tuneCallBack=TuneReportCallback(
    #     {
    #         "mean_val_loss": "mean_val_loss",
    #         "mean_val_acc": "mean_val_acc"
    #     },
    #     on="validation_end")

    #strategy = RayStrategy(num_workers=num_workers,num_cpus_per_worker=num_cpus_per_worker,  use_gpu=True)#num_cpus_per_worker=1, num_workers
    #strategy = RayShardedStrategy(num_workers=1, num_cpus_per_worker=num_cpus_per_worker, use_gpu=True)


    # trainer = pl.Trainer(
    #     #accelerator="cpu", #TODO(remove)
    #     #max_epochs=max_epochs,
    #     #gpus=1,
    #     #precision=experiment.get_parameter("precision"), 
    #     callbacks=[ checkPointCallback ],# TODO unhash,#early_stopping
    #     logger=comet_logger,
    #     # accelerator='auto',
    #     # devices='auto',       
    #     default_root_dir= default_root_dir,
    #     #auto_scale_batch_size="binsearch",
    #     auto_lr_find=True,
    #     check_val_every_n_epoch=10,
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     gradient_clip_val=gradient_clip_val,# 0.5,2.0
    #     log_every_n_steps=2,
    #     strategy=strategy#'ddp'#'ddp' # for multi gpu training
    # )
    #callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc") ]#checkPointCallback
    # callbacks=[early_stopping ]#checkPointCallback
    callbacks=[]#checkPointCallback
    #cuda_now = int(os.environ['cuda_now'])
    
    kwargs = {
        "accelerator":'auto',
         "devices": 'auto',#[cuda_now],
        "max_epochs": max_epochs,
        "callbacks" :callbacks,
        "logger" : comet_logger,
        "default_root_dir" : default_root_dir,
        "auto_lr_find" : False,
        "check_val_every_n_epoch" : 10,
        "accumulate_grad_batches" : accumulate_grad_batches,
        "gradient_clip_val" :gradient_clip_val,
        "log_every_n_steps" :2,
        #"strategy" :'dp',# "ddp_sharded"
        #"profiler":'simple'
        }

    # if os.path.exists(os.path.join(checkpoint_dir, "checkpointtt")):
    #     kwargs["resume_from_checkpoint"] = os.path.join(
    #         checkpoint_dir, "checkpointtt")

    trainer = pl.Trainer(**kwargs)


    #stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging()
    # trainer = pl.Trainer(
    #     #accelerator="cpu", #TODO(remove)
    #     #max_epochs=max_epochs,
    #     #gpus=1,
    #     #precision=experiment.get_parameter("precision"), 
    #     callbacks=[ checkPointCallback ],# TODO unhash,#early_stopping
    #     logger=comet_logger,
    #     # accelerator='auto',
    #     # devices='auto',       
    #     default_root_dir= "/mnt/disks/sdb/lightning_logs",
    #     #auto_scale_batch_size="binsearch",
    #     auto_lr_find=True,
    #     check_val_every_n_epoch=10,
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     gradient_clip_val=gradient_clip_val,# 0.5,2.0
    #     log_every_n_steps=2,
    #     strategy=strategy#'ddp'#'ddp' # for multi gpu training
    # )
    #setting batch size automatically
    #TODO(unhash)
    #trainer.tune(model, datamodule=data)

    trainer.logger._default_hp_metric = False
    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)



 