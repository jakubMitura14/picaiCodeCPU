import monai
from monai.transforms import (
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    AddChanneld,
    Spacingd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Resize,
    Resized,
    RandSpatialCropd,
        AsDiscrete,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    SelectItemsd,
    Invertd,
    DivisiblePadd,
    SpatialPadd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandRicianNoised,
    RandFlipd,
    RandAffined,
    ConcatItemsd,
    RandCoarseDropoutd,
    AsDiscreted,
    MapTransform,
    ResizeWithPadOrCropd
    
)
from monai.config import KeysCollection

import torchio
import numpy as np

class standardizeLabels(MapTransform):

    def __init__(
        self,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0.5).astype('int8')
        return d




def decide_if_whole_image_train(is_whole_to_train, chan3Name,labelName):
    """
    if true we will trian on whole images otherwise just on 64x64x32
    randomly cropped parts
    """
    fff=not is_whole_to_train
    print(f" in decide_if_whole_image_train {fff}")
    if(not is_whole_to_train):
        return [RandCropByPosNegLabeld(
                keys=[chan3Name,labelName],
                label_key=labelName,
                spatial_size=(96, 96, 32),
                pos=1,
                neg=1,
                num_samples=2,
                image_key=chan3Name,
                image_threshold=0
            )
             ]
    return []         


#https://github.com/DIAGNijmegen/picai_prep/blob/19a0ef2d095471648a60e45e30b218b7a81b2641/src/picai_prep/preprocessing.py


def get_train_transforms(RandGaussianNoised_prob
    ,RandAdjustContrastd_prob
    ,RandGaussianSmoothd_prob
    ,RandRicianNoised_prob
    ,RandFlipd_prob
    ,RandAffined_prob
    ,RandCoarseDropoutd_prob
    ,is_whole_to_train
    ,spatial_size
     ):


     
    train_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label"]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label"]),
            AsDiscreted(keys=["label"],to_onehot=2),
            ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label"], spatial_size=spatial_size,),
            ConcatItemsd(keys=["t2w","adc","hbv","adc" ],name="chan3_col_name"),

            standardizeLabels(keys=["label"]),
            #torchio.transforms.OneHot(include=["label"] ), #num_classes=3,
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
            # SelectItemsd(keys=["chan3_col_name","label"]),
            #DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,            
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label"],spatial_size=centerCropSize ),
          
            #*decide_if_whole_image_train(False,"chan3_col_name","label"),
            #SpatialPadd(keys=["chan3_col_name","label"]],spatial_size=maxSize) ,            
            RandGaussianNoised(keys=["chan3_col_name"], prob=RandGaussianNoised_prob),
            RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
            RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
            RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
            RandFlipd(keys=["chan3_col_name","label"], prob=RandFlipd_prob),
            RandAffined(keys=["chan3_col_name","label"], prob=RandAffined_prob),
            RandCoarseDropoutd(keys=["chan3_col_name"], prob=RandCoarseDropoutd_prob,holes=6, spatial_size=5),
            
        ]
    )
    return train_transforms
def get_val_transforms(is_whole_to_train,spatial_size):
    print(f"spatial_sizeeee {spatial_size}"  )

    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label_name_val"]),
            AsDiscreted(keys=["label_name_val"], to_onehot=2),
            ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label_name_val"], spatial_size=spatial_size,),
            ConcatItemsd(["t2w","adc","hbv","adc" ], "chan3_col_name_val"),
            standardizeLabels(keys=["label_name_val"]),

            #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["chan3_col_name","label"]]),
            #Orientationd(keys=["chan3_col_name","label"]], axcodes="RAS"),
            # Spacingd(keys=["chan3_col_name","label"]], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
            #DivisiblePadd(keys=["chan3_col_name_val","label_name_val"],k=32) ,
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label_name_val"],spatial_size=centerCropSize ),

            #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name_val","label_name_val"),
            #SelectItemsd(keys=["chan3_col_name","label"]),
            # ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name")
        ]
    )
    return val_transforms





