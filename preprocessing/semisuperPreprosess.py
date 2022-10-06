#based on https://github.com/DIAGNijmegen/picai_baseline/blob/bb887a32a105e2e38dbcef5e559e7c69c06bc952/src/picai_baseline/nnunet_semi_supervised/generate_automatic_annotations.py#L85

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
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from os import path as pathOs

import numpy as np
import pandas as pd
from functools import partial


@dataclass
class Gleason_score(object):
    """Gleason score with pattern1 + pattern2 = total"""
    pattern1: int
    pattern2: int

    def __post_init__(self):
        assert isinstance(self.pattern1, int) and self.pattern1 >= 0, f"Got invalid pattern1: {self.pattern1}"
        assert isinstance(self.pattern2, int) and self.pattern2 >= 0, f"Got invalid pattern2: {self.pattern2}"

    @property
    def total(self):
        return self.pattern1 + self.pattern2

    @property
    def GGG(self):
        if self.pattern1 == 0 and self.pattern2 == 0:
            return 0
        elif self.pattern1 + self.pattern2 >= 9:
            return 5
        elif self.pattern1 + self.pattern2 == 8:
            return 4
        elif self.pattern1 + self.pattern2 == 7:
            return 3 if self.pattern1 == 4 else 2
        elif self.pattern1 + self.pattern2 <= 6:
            return 1

    def __repr__(self):
        return f"Gleason score({self.pattern1}+{self.pattern2}={self.total})"

    def __str__(self):
        return self.__repr__()

    def to_tuple(self):
        return self.pattern1, self.pattern2, self.total

"""
analyzing data about gleason from metadata to get idea how many lesions should be rocegnized 
in the image
"""
def get_numb_ofLesions_toRetain(row):
    row = row[1]
    # grab lesion Gleason scores scores
    gleason_scores = []
    if isinstance(row['lesion_GS'], float) and np.isnan(row['lesion_GS']):
        gleason_scores = []
    else:
        gleason_scores = row['lesion_GS'].split(",")

    # convert Gleason scores to ISUP grades
    isup_grades = []
    for score in gleason_scores:
        if score == "N/A":
            continue

        pattern1, pattern2 = score.split("+")
        GS = Gleason_score(int(pattern1), int(pattern2))
        isup_grades.append(GS.GGG)
    return sum([score >= 2 for score in isup_grades])

"""
iterates over all metadata and saves the number of lesions we are intrested to retain
"""
def iterate_and_addLesionNumber(df):
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(get_numb_ofLesions_toRetain ,list(df.iterrows())) 
    df["num_lesions_to_retain"]=resList
    return df


"""
creates the file with dummy data for label - in cases where is no valid labels
outPath - full directory together with file name and extension
dim_ x,y,z dimensions of the label we want
imageRef_path - reference image from which we will take metadata
"""
def writeDummyLabels(outPath,imageRef_path):
    origImage = sitk.ReadImage(imageRef_path)
    #intentionally inverting order as it is expected by simple itk
    data= sitk.GetArrayFromImage(origImage)
    sizz= data.shape
    print(f"sizz {sizz} ")
    arr= np.zeros((sizz[0],sizz[1],sizz[2])).astype('int32')
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing(origImage.GetSpacing())
    image.SetOrigin(origImage.GetOrigin())
    image.SetDirection(origImage.GetDirection())
    #saving to hard drive
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(outPath)
    writer.Execute(image)  
    return sizz  