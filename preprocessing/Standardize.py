# pathBaselineImage ='/mnt/disks/sdb/10001/10001_1000001_t2w.mha'
from __future__ import print_function
import collections
import functools
import math
import multiprocessing as mp
import time
from functools import partial
from os import listdir
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
import numpy as np
import pandas as pd
import SimpleITK as sitk
from intensity_normalization.normalize.nyul import NyulNormalize


def removeOutliersBiasFieldCorrect(path,numberOfStandardDeviations = 4):
    """
    all taken from https://github.com/NIH-MIP/Radiology_Image_Preprocessing_for_Deep_Learning/blob/main/Codes/Main_Preprocessing.py
    my modification that instead of histogram usage for outliers I use the standard deviations
    path - path to file to be processed
    numberOfStandardDeviations- osed to define outliers

    """
    
    image = sitk.ReadImage(path)
    print("newwD")
    
#     data = sitk.GetArrayFromImage(image1)
#     # shift the data up so that all intensity values turn positive
#     stdd = np.std(data)*5
#     median = np.median(data)
#     data = np.clip(data, median-numberOfStandardDeviations*stdd, median+numberOfStandardDeviations*stdd)
#     data -= np.min(data)
#     #TO normalize an image by mapping its [Min,Max] into the interval [0,255]
#     N=255
#     data=N*(data+600)/2000

#     #recreating image keeping relevant metadata
#     image = sitk.GetImageFromArray(data)
#     image.SetSpacing(image1.GetSpacing())
#     image.SetOrigin(image1.GetOrigin())
#     image.SetDirection(image1.GetDirection())
    #bias field normalization
    # maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    inputImage = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # numberFittingLevels = 4
    imageB = corrector.Execute(inputImage)
    # imageB.SetSpacing(image.GetSpacing())
    # imageB.SetOrigin(image.GetOrigin())
    # imageB.SetDirection(image.GetDirection())
    return imageB


def removeOutliersAndWrite(path):
    outPath = path.replace('.mha','_bfc.mha')
    print("biasFieldCorrect "+path)

    if(not pathOs.exists(outPath)):
        image=removeOutliersBiasFieldCorrect(path)

        #standardazing orientation 
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(outPath)
        writer.Execute(image)
    return outPath       
    

def standardizeFromPathAndOverwrite(path,nyul_normalizer): 
    #print("standardizeFromPathAndOverwrite "+path)
    newPath = path#.replace(".mha","_stand.mha" )
    
    if(pathOs.exists(newPath)):
        return newPath
    
    image1=sitk.ReadImage(path)
    image1 = sitk.DICOMOrient(image1, 'RAS')
    image1 = sitk.Cast(image1, sitk.sitkFloat32)
    data=nyul_normalizer(sitk.GetArrayFromImage(image1))
    #recreating image keeping relevant metadata
    image = sitk.GetImageFromArray(data)  
    image.SetSpacing(image1.GetSpacing())
    image.SetOrigin(image1.GetOrigin())
    image.SetDirection(image1.GetDirection())    
    
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(image)    
    return newPath


def denoise(path):
    image1=sitk.ReadImage(path)
    imgSmooth = sitk.CurvatureFlow(image1=image1, timeStep=0.125,numberOfIterations=5)
    #standardazing orientation
    image1 = sitk.Cast(image1, sitk.sitkFloat32)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(path)
    writer.Execute(imgSmooth)   

def getMedianCorner(image1):
    """
    get median from 4 corners as value for background
    """
    sizz=image1.GetSize()
    sizzMinus=(sizz[0]-1,sizz[1]-1,sizz[2]-1)
    corners=[(0,0,0),(0,0,sizzMinus[2]), (0,sizzMinus[1],0),(sizzMinus[0],0,0) 
            ,(0,sizzMinus[1],sizzMinus[2]),(sizzMinus[0],0,sizzMinus[2]),(sizzMinus[0],sizzMinus[1],0)
            ,(sizzMinus[0],sizzMinus[1],sizzMinus[2])  ]

    cornerValues=list(map(lambda coords: image1.GetPixel(coords)  ,corners ))
    return np.median(cornerValues)


def padToSize(image1,targetSize, paddValue):
    """
    padd with given value symmetrically to get the predifined target size and return padded image
    """
    currentSize=image1.GetSize()
    sizediffs=(targetSize[0]-currentSize[1]  , targetSize[1]-currentSize[1]  ,targetSize[2]-currentSize[2])
    halfDiffSize=(math.floor(sizediffs[0]/2) , math.floor(sizediffs[1]/2), math.floor(sizediffs[2]/2))
    rest=(sizediffs[0]-halfDiffSize[0]  ,sizediffs[1]-halfDiffSize[1]  ,sizediffs[2]-halfDiffSize[2]  )
    
    #print(f" currentSize {currentSize} targetSize {targetSize} halfDiffSize {halfDiffSize}  rest {rest} paddValue {paddValue} sizediffs {type(sizediffs)}")
    
    halfDiffSize=np.array(halfDiffSize, dtype='int').tolist() 
    rest=np.array(rest, dtype='int').tolist() 
    
    #saving only non negative entries
    halfDiffSize_to_pad= list(map(lambda dim : max(dim,0) ,halfDiffSize ))
    rest_to_pad= list(map(lambda dim : max(dim,0) ,rest ))
    #get only negative entries - those mean that we need to crop and we negate it to get positive numbers
    halfDiffSize_to_crop= list(map(lambda dim : (-1)*min(dim,0) ,halfDiffSize ))
    rest_to_crop= list(map(lambda dim : (-1)*min(dim,0) ,rest ))

    padded= sitk.ConstantPad(image1, halfDiffSize_to_pad, rest_to_pad, paddValue)
    res= sitk.Crop(padded, halfDiffSize_to_crop,rest_to_crop )
    print(f"result size {res.GetSize()} target size {targetSize}")
    
    return res
    #return sitk.ConstantPad(image1, (1,1,1), (1,1,1), paddValue)


def padToDivisibleBy32(image1,paddValue):
    """
    padds so all dimensions will be divisible by 32
    """


    sizz=image1.GetSize()
    targetSize=(math.ceil(sizz[0]/32)*32, math.ceil(sizz[1]/32)*32,math.ceil(sizz[2]/32)*32  )
    return padToSize(image1,targetSize, paddValue)
    

def padToAndSaveLabel(row,colname,targetSize, paddValue,keyword,isTobeDiv):
    row=row[1]
    writer = sitk.ImageFileWriter()
    path = str(row[colname])
    if(path!=" "):
        outPath = path.replace('.nii.gz',keyword+ 'padded'+'.nii.gz')
        image=sitk.ReadImage(str(path))
        # print(f"resize label too {targetSize}")
        # data= sitk.GetArrayFromImage(image)

        #print(f"unique in label {np.unique(data)}")

        # if(isTobeDiv):
        #     image=padToDivisibleBy32(image,paddValue)
        #     writer.KeepOriginalImageUIDOn()
        #     writer.SetFileName(outPath)
        #     writer.Execute(image) 
        # else:
        image=padToSize(image,targetSize,paddValue)
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(outPath)
        writer.Execute(image)

        print(f"result label size {image.GetSize()}  target size {targetSize}")
        return outPath 
    return " "                     
                


#######  


def iterateAndpadLabels(df,colname,targetSize, paddValue,keyword,isTobeDiv):
    reslist=[]
    
    with mp.Pool(processes = mp.cpu_count()) as pool:
        reslist=pool.map(partial(padToAndSaveLabel,colname=colname,targetSize=targetSize,paddValue=paddValue,keyword=keyword,isTobeDiv=isTobeDiv ),list(df.iterrows()))
    #reslist=list(map(partial(padToAndSaveLabel,colname=colname,targetSize=targetSize,paddValue=paddValue,keyword=keyword,isTobeDiv=isTobeDiv ),list(df.iterrows())))


    df["label"+keyword]=reslist


def iterateAndBiasCorrect(seriesString,df):
    train_patientsPaths=df[seriesString].dropna().astype('str').to_numpy()
    train_patientsPaths=list(filter(lambda path: len(path)>2 ,train_patientsPaths))   
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(removeOutliersAndWrite,train_patientsPaths)
    df['bfc_'+seriesString]=resList      



def iterateAndDenoise(seriesString,df):
    train_patientsPaths=df[seriesString].dropna().astype('str').to_numpy()
    train_patientsPaths=list(filter(lambda path: len(path)>2 ,train_patientsPaths))   
    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(denoise,train_patientsPaths)


def iterateAndStandardize(seriesString,df,trainedModelsBasicPath,numberOfSamples):
    """
    iterates over files from train_patientsPaths representing seriesString type of the study
    and overwrites it with normalised biased corrected and standardised version
    intensity_normalization_modality - what type of modality we are normalizing
        from https://github.com/jcreinhold/intensity-normalization/blob/03dbdc84bfbb35623ae1920b802d37f5c8056658/intensity_normalization/typing.py
        FLAIR: builtins.str = "flair"
        MD: builtins.str = "md"
        OTHER: builtins.str = "other"
        PD: builtins.str = "pd"
        T1: builtins.str = "t1"
        T2: builtins.str = "t2"
    """
    train_patientsPaths=df[seriesString].dropna().astype('str').to_numpy()
    train_patientsPaths=list(filter(lambda path: len(path)>2 ,train_patientsPaths))
        
    print("fitting normalizer  " +seriesString)
    nyul_normalizer = NyulNormalize()
    #we need to avoid getting too much into normalizer becouse it will lead to memory errors
    randomPart=[]
    if(len(train_patientsPaths)<numberOfSamples):
        randomPart=train_patientsPaths
    else:
        randomPart = np.random.choice(train_patientsPaths,numberOfSamples , replace=False)
    
    images = [sitk.GetArrayFromImage(sitk.ReadImage(image_path)) for image_path in randomPart]  
    nyul_normalizer.fit(images)

    pathToSave = join(trainedModelsBasicPath,seriesString+".npy")
    nyul_normalizer.save_standard_histogram(pathToSave)
    #reloading from disk just for debugging
    nyul_normalizer.load_standard_histogram(pathToSave)
    
    results=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        results=pool.map(partial(standardizeFromPathAndOverwrite,nyul_normalizer=nyul_normalizer ),train_patientsPaths)
    # colName= 'Nyul_'+seriesString
    # df[colName]=toUp 
    return results

#Important !!! set all labels that are non 0 to 1
def changeLabelToOnes(row):
    """
    as in the labels or meaningfull ones are greater then 0 so we need to process it and change any nymber grater to 0 to 1...
    """
    row=row[1]
    path=row['reSampledPath']
    path_t2w=row['t2w']
    if(path!= " " and path!=""):
        image1 = sitk.ReadImage(path)
        image1 = sitk.DICOMOrient(image1, 'RAS')
        #image1 = sitk.Cast(image1, sitk.sitkFloat32)
        data = sitk.GetArrayFromImage(image1)
        data = (data > 0.5).astype('int32')
        print(f" at begining unique   {np.unique(data)}"  )
        #recreating image keeping relevant metadata
        image = sitk.GetImageFromArray(data)
        image.SetSpacing(image1.GetSpacing())
        image.SetOrigin(image1.GetOrigin())
        image.SetDirection(image1.GetDirection())
        #standardazing orientation
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        newPath= path_t2w.replace(".mha","_stand_label.mha")
        writer.SetFileName(newPath)
        writer.Execute(image)   
    
def iterateAndchangeLabelToOnes(df):
    """
    iterates over files from train_patientsPaths representing seriesString type of the study
    and overwrites it with normalised biased corrected and standardised version
    """
    #paralelize https://medium.com/python-supply/map-reduce-and-multiprocessing-8d432343f3e7
    # train_patientsPaths=df['reSampledPath'].dropna().astype('str').to_numpy()
    # train_patientsPaths=list(filter(lambda path: len(path)>2 ,train_patientsPaths))
    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(changeLabelToOnes,list(df.iterrows()))
    return df
    # toUp=np.full(df.shape[0], False)#[0:3]=[True,True,True]
    # toUp[0:numRows]=np.full(numRows, True)
    #df['labels_to_one']=toUp    
