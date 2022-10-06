import os
from copy import deepcopy
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
from KevinSR import SOUP_GAN, mask_interpolation



def copyDirAndOrigin(imageOrig,spacing,data):
    image1 = sitk.GetImageFromArray(data)
    image1.SetSpacing(spacing) #updating spacing
    image1.SetOrigin(imageOrig.GetOrigin())
    image1.SetDirection(imageOrig.GetDirection()) 
    #print(image1.GetSize())
    return image1

def useGan(data,post_slices,pre_slices,stepSize):
    diffPrim = post_slices-pre_slices# always positive number if too big we will run out of memoory
    diffDone=0
    #print(f"sstep size {stepSize}")
    #stepSize=100 # how big diffrence in number of slices it can handle at one go
    if(diffPrim<stepSize):
            return SOUP_GAN(data, post_slices/pre_slices,1)
    else:        
        for st in range(stepSize,diffPrim,stepSize):
            Z_FAC = (pre_slices+st)/(pre_slices+st-stepSize)
            diffDone=st
            data = SOUP_GAN(data, Z_FAC,1)
            
    return SOUP_GAN(data, post_slices/(pre_slices+diffDone),1)
 
#pathT2w,pathHbv,pathADC,patht2wLabel
def resample_with_GAN(path, targetSpac):
    imageOrig = sitk.ReadImage(path)
    origSize= imageOrig.GetSize()
    orig_spacing=imageOrig.GetSpacing()
    currentSpacing = list(orig_spacing)
    print(f"origSize {origSize}")
    #new size of the image after changed spacing
    new_size = tuple([int(origSize[0]*(orig_spacing[0]/targetSpac[0])),
                    int(origSize[1]*(orig_spacing[1]/targetSpac[1])),
                    int(origSize[2]*(orig_spacing[2]/targetSpac[2]) )  ]  )
    print(f"new_size {new_size}  target spacing {targetSpac}")
    anySuperSampled = False
    data=sitk.GetArrayFromImage(imageOrig)
    #supersampling if needed

    # for axis in [0,1,2]:   
    #     if(new_size[axis]>origSize[axis]):
    #         anySuperSampled=True
    #         #in some cases the GPU memory is not cleared enough
    #         #device = cuda.get_current_device()
    #         #device.reset()
    #         currentSpacing[axis]=targetSpac[axis]
    #         pre_slices = origSize[axis]
    #         post_slices = new_size[axis]
    #         Z_FAC = post_slices/pre_slices # Sampling factor in Z direction
    #         if(axis==1):
    #             data = np.moveaxis(data, 1, 2)
    #         if(axis==2):
    #             data = np.moveaxis(data, 0, 2)
    #         #Call the SR interpolation tool from KevinSR
    #         #print(f"thicks_ori shape {data.shape} ")
    #         data =useGan(data,post_slices,pre_slices,100)

    #         # try:
    #         #     data =useGan(data,post_slices,pre_slices,200)
    #         # except Exception as e:
    #         #     print(e)
    #         #     try:
    #         #        data =useGan(data,post_slices,pre_slices,100) 
    #         #     except Exception as e:
    #         #         print(e)    
    #         #         try: 
    #         #             data =useGan(data,post_slices,pre_slices,50) 
    #         #         except Exception as e:
    #         #             print(e)        
    #         #             data =useGan(data,post_slices,pre_slices,25) 


    #         #data = SOUP_GAN(data, Z_FAC,1)
    #         #print(f"thins_gen shape {data.shape} ")
    #         if(axis==1):
    #             data = np.moveaxis(data, 2, 1)
    #         if(axis==2):
    #             data = np.moveaxis(data, 2, 0)            
            

    #we need to recreate itk image object only if some supersampling was performed
    if(anySuperSampled):
        image=copyDirAndOrigin(imageOrig,tuple(currentSpacing),data)
        print(f"sssssupersampled {image.GetSize()}")
    else:
        image=imageOrig
    #copmpleting resampling given some subsampling needs to be performed
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetSize(new_size)
    res= resample.Execute(image)
    
    return res
    
#pathT2w,pathHbv,pathADC,patht2wLabel
def resample_label_with_GAN(path, targetSpac):
    """
    nearly the same as resample_with_GAN but in downsampling we get nearest neighbour interpolator and in GAN specialized label interpolator
    """
    imageOrig = sitk.ReadImage(path)
    origSize= imageOrig.GetSize()
    orig_spacing=imageOrig.GetSpacing()
    currentSpacing = list(orig_spacing)
    print(f"origSize {origSize}")
    #new size of the image after changed spacing
    new_size = tuple([int(origSize[0]*(orig_spacing[0]/targetSpac[0])),
                    int(origSize[1]*(orig_spacing[1]/targetSpac[1])),
                    int(origSize[2]*(orig_spacing[2]/targetSpac[2]) )  ]  )
    print(f"new_size {new_size}")
    anySuperSampled = False
    data=sitk.GetArrayFromImage(imageOrig)
    data = (data > 0.5).astype('int8')

    print(f" at resampling unique   {np.unique(data)}"  )
   
    #supersampling if needed
    
    # for axis in [0,1,2]:   
    #     if(new_size[axis]>origSize[axis]):
    #         anySuperSampled=True
    #         #in some cases the GPU memory is not cleared enough
    #         #device = cuda.get_current_device()
    #         #device.reset()
    #         currentSpacing[axis]=targetSpac[axis]
    #         pre_slices = origSize[axis]
    #         post_slices = new_size[axis]
    #         Z_FAC = post_slices/pre_slices # Sampling factor in Z direction
    #         if(axis==1):
    #             data = np.moveaxis(data, 1, 2)
    #         if(axis==2):
    #             data = np.moveaxis(data, 0, 2)
    #         #Call the SR interpolation tool from KevinSR
    #         #print(f"thicks_ori shape {data.shape} ")

    #         data = mask_interpolation(data, Z_FAC)
    #         #print(f"thins_gen shape {data.shape} ")
    #         if(axis==1):
    #             data = np.moveaxis(data, 2, 1)
    #         if(axis==2):
    #             data = np.moveaxis(data, 2, 0)            
            

    # #we need to recreate itk image object only if some supersampling was performed
    # if(anySuperSampled):
    #     image=copyDirAndOrigin(imageOrig,tuple(currentSpacing),data)
    # else:
    #     image=imageOrig

    image=copyDirAndOrigin(imageOrig,tuple(currentSpacing),data)
    
    #image=imageOrig

    #copmpleting resampling given some subsampling needs to be performed
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetSize(new_size)
    res= resample.Execute(image)
    res = sitk.DICOMOrient(res, 'RAS')


    return res
