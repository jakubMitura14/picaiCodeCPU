import functools
import multiprocessing as mp
import os
from functools import partial
from zipfile import BadZipFile, ZipFile
import numpy as np
import pandas as pd
import SimpleITK as sitk

#read metadata, and add columns for additional information
csvPath='/mnt/disks/sdb/labels/clinical_information/marksheet.csv'
df = pd.read_csv(csvPath)
#initializing empty columns
df["reSampledPath"] = ""
df["adc"] = ""
df["cor"] = ""
df["hbv"] = ""
df["sag"] = ""
df["t2w"] = ""
df["isAnythingInAnnotated"] = 0
df["isAnyMissing"] = False

df["adc_resmaplA"]=""
df["hbv_resmaplA"]=""
# for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:
#     colName= 'stand_and_bias_'+keyWord
#     df[colName]=False
#     colName= 'Nyul_'+keyWord
#     df[colName]=False
df['labels_to_one']=False 

targetDir= '/mnt/disks/sdb/orig'
def unpackk(zipDir,targetDir):
    with ZipFile(zipDir, "r") as zip_ref:
        for name in zip_ref.namelist():
            #ignoring all corrupt files
            try:
                zip_ref.extract(name, targetDir)
            except BadZipFile as e:
                print(e)
    
    
    # with ZipFile(zipDir, 'r') as zipObj:
    #    # Extract all the contents of zip file in different directory
    #    zipObj.extractall(targetDir)
        
# unpackk( '/mnt/disks/sdb/picai_public_images_fold0.zip', targetDir)      
# unpackk( '/mnt/disks/sdb/picai_public_images_fold1.zip', targetDir)      
# unpackk( '/mnt/disks/sdb/picai_public_images_fold2.zip', targetDir)      
# unpackk( '/mnt/disks/sdb/picai_public_images_fold3.zip', targetDir)      
# unpackk( '/mnt/disks/sdb/picai_public_images_fold4.zip', targetDir) 

unpackk( '/mnt/disks/sdb/origB/picai_public_images_fold0.zip', targetDir)      
unpackk( '/mnt/disks/sdb/origB/picai_public_images_fold1.zip', targetDir)      
unpackk( '/mnt/disks/sdb/origB/picai_public_images_fold2.zip', targetDir)      
unpackk( '/mnt/disks/sdb/origB/picai_public_images_fold3.zip', targetDir)      
unpackk( '/mnt/disks/sdb/origB/picai_public_images_fold4.zip', targetDir) 



dirDict={}
for subdir, dirs, files in os.walk(targetDir):
    for subdirin, dirsin, filesin in os.walk(subdir):
        lenn= len(filesin)
        if(lenn>0):
            try:
                dirDict[subdirin.split("/")[5]]=filesin
            except:
                pass
            # print(f"subdir {subdir}")
            #print(subdir.split("/"))
            # print(subdirin.split("/"))
            #dirDict[subdir]=filesin

print(dirDict)

labelsFiles=[]
labelsRootDir = '/mnt/disks/sdb/labels/csPCa_lesion_delineations/human_expert/resampled/'
for subdir, dirs, files in os.walk(labelsRootDir):
    labelsFiles=files
    
#Constructing functions that when applied to each row will fill the necessary path data
listOfDeficientStudyIds=[]

print(dirDict)


#create a dictionary of directories where key is the patient_id
def findPathh(row,dirDictt,keyWord,targetDir):
    row=row[1]
    patId=str(row['patient_id'])
    study_id=str(row['study_id'])
    #first check is such key present
    if(patId in dirDictt ):
        filtered = list(filter(lambda file_name:   (keyWord in file_name and  study_id  in  file_name  ), dirDictt[patId]  ))
        if(len(filtered)>0):
            return os.path.join(targetDir,  patId, filtered[0] )
        else:
            print(f"no {keyWord} in {study_id}")
            listOfDeficientStudyIds.append(study_id)
            return " " 
    listOfDeficientStudyIds.append(study_id)
    return " "

def iter_paths_apply(dff,keyword):
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(findPathh,dirDictt=dirDict,keyWord=keyword,targetDir=targetDir)  ,list(dff.iterrows()))
    dff[keyword]=resList   

iter_paths_apply(df,'t2w')
iter_paths_apply(df,'adc')
iter_paths_apply(df,'hbv')
iter_paths_apply(df,'sag')
iter_paths_apply(df,'cor')


def findResampledLabel(row,labelsFiles):
    patId=str(row['patient_id'])
    study_id=str(row['study_id'])
    filtered = list(filter(lambda file_name:   (study_id  in  file_name  ), labelsFiles ))
    if(len(filtered)>0):
        return os.path.join(labelsRootDir, filtered[0])
    listOfDeficientStudyIds.append(study_id)    
    return " "
    
        
df["reSampledPath"] =  df.apply(lambda row : findResampledLabel(row,labelsFiles )   , axis = 1)  

def isAnythingInAnnotated(row):
    row=row[1]
    reSampledPath=str(row['reSampledPath'])
    if(len(reSampledPath)>1):
        image = sitk.ReadImage(reSampledPath)
        nda = sitk.GetArrayFromImage(image)
        return np.sum(nda)
    return 0

resList=[]
with mp.Pool(processes = mp.cpu_count()) as pool:
    resList=pool.map(isAnythingInAnnotated  ,list(df.iterrows()))
df['isAnythingInAnnotated']=resList   

#marking that we have something lacking here
df["isAnyMissing"]=df.apply(lambda row : str(row['study_id']) in  listOfDeficientStudyIds  , axis = 1) 

def ifShortReturnMinus(tupl, patId,colName):
    if(len(tupl)!=3):
        print("incorrect spacial data "+ str(colName)+ "  "+str(patId)+ " length "+ str(len(tupl))+ "  "+ str(tupl) ) 
        return (-1,-1,-1)
    return tupl    
#getting sizes and spacings ...
def get_spatial_meta(row,colName):
    row=row[1]
    patId=str(row['patient_id'])
    path=str(row[colName])
    if(len(path)>1):
        image = sitk.ReadImage(path)
        sizz= ifShortReturnMinus(image.GetSize(),patId,colName )
        spac= ifShortReturnMinus(image.GetSpacing(),patId,colName)
        orig= ifShortReturnMinus(image.GetOrigin(),patId,colName)
        return list(sizz)+list(spac)+list(orig)
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1]
for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:    
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(get_spatial_meta,colName=keyWord)  ,list(df.iterrows()))    
    print(type(resList))    
    df[keyWord+'_sizz_x']= list(map(lambda arr:arr[0], resList))    
    df[keyWord+'_sizz_y']= list(map(lambda arr:arr[1], resList))    
    df[keyWord+'_sizz_z']= list(map(lambda arr:arr[2], resList))

    df[keyWord+'_spac_x']= list(map(lambda arr:arr[3], resList))    
    df[keyWord+'_spac_y']= list(map(lambda arr:arr[4], resList))    
    df[keyWord+'_spac_z']= list(map(lambda arr:arr[5], resList))    
    
    df[keyWord+'_orig_x']= list(map(lambda arr:arr[6], resList))    
    df[keyWord+'_orig_y']= list(map(lambda arr:arr[7], resList))    
    df[keyWord+'_orig_z']= list(map(lambda arr:arr[8], resList))    





df.to_csv('/mnt/disks/sdb/metadata/processedMetaData.csv') 
