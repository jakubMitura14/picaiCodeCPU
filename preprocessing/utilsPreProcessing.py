import SimpleITK as sitk

def write_to_modif_path(image, path,toRemove,newString):
    """
    get path removes toRemove from end and replaces it with newString
    to newly created path writes image
    """
    newPath= path.replace(toRemove,newString)
    writer = sitk.ImageFileWriter()
    #writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(image)   