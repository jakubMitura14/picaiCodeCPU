import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys
import os.path
from os import path as pathOs
import comet_ml
from comet_ml import Experiment
import numpy as np



def reg_adc_hbv_to_t2w_sitk(row,colName,t2wColName,outPathh=""):
    """
    registers adc and hbv images to t2w image
    first we need to create directories for the results
    then we suply hbv and adc as moving images and t2w as static one - and register in reference to it
    we do it in multiple threads at once and we waiteach time the process finished
    """

    study_id=str(row[1]['study_id'])
    
    patId=str(row[1]['patient_id'])
    print(patId)
    path=str(row[1][colName])
    outPath=outPathh
    if(outPath==""):
        outPath = path.replace(".mha","_reg.mha")
    #returning faster if the result is already present
    if(pathOs.exists(outPath)):
        return outPath 
        #pass     
    else:
        if(len(path)>2):
            #creating the folder if none is present
            fixed_image = sitk.ReadImage(row[1][t2wColName])
            moving_image = sitk.ReadImage(path)
            fixed_image=sitk.Cast(fixed_image, sitk.sitkFloat32)
            moving_image=sitk.Cast(moving_image, sitk.sitkFloat32)

            #sitk euler    
            reg_image=euler_sitk(fixed_image, moving_image)

            #save
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(outPath)
            writer.Execute(reg_image)   
            #we will repeat operation multiple max 9 times if the result would not be written
            return outPath            
        else:
            return ""    
    return ""

def euler_sitk(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.MOMENTS)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.8)
    registration_method.SetInterpolator(sitk.sitkBSpline)

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # The order of parameters for the Euler3DTransform is [angle_x, angle_y, angle_z, t_x, t_y, t_z]. The parameter 
    # sampling grid is centered on the initial_transform parameter values, that are all zero for the rotations. Given
    # the number of steps, their length and optimizer scales we have:
    # angle_x = 0
    # angle_y = -pi, 0, pi
    # angle_z = -pi, 0, pi
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,numberOfIterations=1, estimateLearningRate = registration_method.EachIteration)
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=[0,1,1,0,0,0], stepLength = np.pi, numberOfIterations=1000)
    # registration_method.SetOptimizerScales([1,1,1,1,1,1])

    #Perform the registration in-place so that the initial_transform is modified.
    outTx=registration_method.SetInitialTransform(initial_transform, inPlace=True) 
    
    outTx=registration_method.Execute(fixed_image, moving_image)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving_image)    
    
    return out

def getRegistrationScore(logPath):
    """
    given log path will extract the registration metric
    """
    #logPath= '/mnt/disks/sdb/orig/10021/10021_1000021_adctw_for_adcb_tw_/elastix.log'
    fp= open(logPath, 'r')
    content=   fp.read()
    len(content)
    lineWithRes= list(filter(lambda line: "Final metric value " in line ,content.split("\n")))
    trimmed= lineWithRes[0].split("=")[1][1:-1]
    return float(trimmed)



def reg_adc_hbv_to_t2w(row,colName,elacticPath,reg_prop,t2wColName,experiment=None,reIndex=0):
    """
    registers adc and hbv images to t2w image
    first we need to create directories for the results
    then we suply hbv and adc as moving images and t2w as static one - and register in reference to it
    we do it in multiple threads at once and we waiteach time the process finished
    """

    study_id=str(row[1]['study_id'])
    
    patId=str(row[1]['patient_id'])
    path=str(row[1][colName])
    outPath = path.replace(".mha","_for_"+colName)
    result=pathOs.join(outPath,"result.0.mha")
    logPath=pathOs.join(outPath,"elastix.log")
    # print(result)
    # print(pathOs.exists(result))
    #returning faster if the result is already present
    #if(pathOs.exists(outPath)):
    if(pathOs.exists(result)):
        if(experiment!=None):
            experiment.log_text(f"already registered {colName} {study_id}")    
        print(f"registered already present {patId}")
        return result     
    else:
        if(len(path)>1):
            #creating the folder if none is present
            if(not pathOs.exists(outPath)):
                cmd='mkdir '+ outPath
                p = Popen(cmd, shell=True)
                p.wait()
            print(f"**********  ***********  ****************  registering {patId}  ")
            #euler_sitk(sitk.ReadImage(row[1][t2wColName]), sitk.ReadImage(path))

            # parameterMap = sitk.GetDefaultParameterMap('translation')
            # parameterMap['MaximumNumberOfIterations'] = ['1']
            # parameterMap['Interpolator'] = ['BSplineInterpolator']
            # resultImage = sitk.Elastix(sitk.ReadImage(row[1][t2wColName]),  \
            #                         sitk.ReadImage(path), \
            #                         parameterMap)
            # writer = sitk.ImageFileWriter()
            # writer.KeepOriginalImageUIDOn()
            # writer.SetFileName(result)
            # writer.Execute(resultImage) 



            cmd=f"{elacticPath} -f {row[1][t2wColName]} -m {path} -out {outPath} -p {reg_prop} -threads 1"
            print(cmd)
            p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
            p.wait()
            #we will repeat operation multiple max 9 times if the result would not be written
            if((not pathOs.exists(result)) and reIndex<8):
                reIndexNew=reIndex+1
                if(reIndex==4): #in case it do not work we will try diffrent parametrization
                    reg_prop=reg_prop.replace("parameters","parametersB")              
                reg_adc_hbv_to_t2w(row,colName,elacticPath,reg_prop,t2wColName,experiment,reIndexNew)
            if(not pathOs.exists(result)):
                print("registration unsuccessfull")
                return " "



            #in case it will not work via elastix we will use simple itk    
            # if(not pathOs.exists(result)):
            #     try:
            #         reg_adc_hbv_to_t2w_sitk(row,colName,t2wColName,result)
            #     except:
            #         #maybe it can not be done ?
            #     return ("",0.0)
            #return (result,getRegistrationScore(logPath) )    #        
            return result #        
        else:
            return " "    
    return " "    
