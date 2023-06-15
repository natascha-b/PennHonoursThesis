import numpy as np
import pandas as pd
from astropy.io.fits import open as fits_open

import os
import tqdm

def getFilepaths(rootDir, extension='.fits', prefix='diff'):
    """
    Generate array of file paths to open files 
    Input:
        rootDir: path of directory containing images (which can be in subdirectories)
        extension: the filetype desired; default == .fits
        prefix: so that we only return one filepath for each ID. Note that this assumes that diff, srch, and 
                tmpl images are in the same directory, and the only difference is the file prefix
        
    Returns:
        array with file paths
    """
    filePaths = []
    for dirpath, dirnames, filenames in os.walk(rootDir):
        for filename in filenames:
            if filename.endswith(extension) and filename.startswith(prefix):
                path = os.path.join(dirpath, filename)
                filePaths.append(path)
    print(f'# of files: {len(filePaths)}')
    return filePaths

def getIDs(filePaths, unique=True):
    """
    Input:
        filePaths: array of file paths to images
        unique: whether you want the unique IDs, or repeated ids for diff tmpl srch triplets (default True)
        
        (in theory this is redundant, since getFilepaths() returns only one prefix type)
    
    Returns:
        list of IDs for images
    """
    if unique:
        return np.unique([int(path.split('/')[-1][4:-5]) for path in filePaths]).tolist()
    
    else:
        return [int(path.split('/')[-1][4:-5]) for path in filePaths]

def batchReadImages(filePaths, featuresDataFrame, startIdx=0, batchSize=25):
    """
    Read images in batches to prevent any code breakage, etc. Output is np.array of images and a list of the 
    associated info (ID, object type)
    
    Input:
        filePaths: file paths, obtained with getFilepaths() function. They MUST be the diff prefix
        featuresDataFrame: pandas df with features, must have ['ID'] and ['OBJECT_TYPE'] columns
        startIdx: where to start from for batches; default = 0 to just run through all images if 
                  number of images is relatively small
        batchSize: size of batches, defaults to 25
        
    Returns:
        batchedImageArray: np.array of shape (batch_size, row, col, 3), in order diff, tmpl, srch
        infoArray: array of tuples, (ID, OBJECT_TYPE)
    """
    
    row, col = fits_open(filePaths[0])[0].data.shape
    batchedImageArray = np.zeros((batchSize, row, col, 3)) #3 refers to diff, tmpl, srch triplet
    infoList = np.zeros((batchSize, 2), dtype=int)
    # IDlist = getIDs(filePaths)
        
    for i, fileName in enumerate(filePaths[startIdx : startIdx + batchSize]):
        # print(fileName) #works fine here
        
        if fileName.split('/')[-1][0:4] != 'diff':
            prefix = fileName.split('/')[-1][0:4]
            fileName.replace(prefix, 'diff')
            print("filePrefix wasn't 'diff'; replaced")
        
        diffhdul = fits_open(fileName)
        diff = diffhdul[0].data
        diffhdul.close()
        
        srchhdul = fits_open(fileName.replace('diff', 'srch'))
        srch = srchhdul[0].data
        srchhdul.close()
        
        tmplhdul = fits_open(fileName.replace('diff', 'temp'))
        tmpl = tmplhdul[0].data
        tmplhdul.close()
        
        batchedImageArray[i, :, :, 0] = np.array(diff)
        batchedImageArray[i, :, :, 1] = np.array(srch)
        batchedImageArray[i, :, :, 2] = np.array(tmpl)
        
        targetID = int(fileName.split('/')[-1][4:-5])
        # targetID = IDlist[i]
        objType = int(featuresDataFrame[featuresDataFrame.ID == targetID].OBJECT_TYPE)
        infoList[i] = (targetID, objType)
        
        if len(str(targetID)) < 3:
            raise RuntimeError('targetID')
    
    return batchedImageArray, infoList

def openBatched(filePaths, featuresDataFrame, startIdx=0, batchSize=25):
    """
    This function loops over batchReadImages() combine images from batches into single output imageArray
    
    Input:
        FilePaths: file paths, obtained with getFilepaths() function. They MUST be the diff prefix
        featuresDataFrame: pandas df with features, must have ['ID'] and ['OBJECT_TYPE'] columns
        startIdx: where to start from for batches; default = 0 to just run through all images if 
                  number of images is relatively small
        batchSize: size of batches, defaults to 25
    Returns:
        imageArray: np.array of shape (batch_size, row, col, 3), in order diff, tmpl, srch
        infoArray: array of tuples, (ID, OBJECT_TYPE)
    """
        
    numImages = len(filePaths)
    if batchSize > numImages: #if batchSize is larger than numImages, reduce to open all at once
        batchSize = numImages
    
    row, col = fits_open(filePaths[0])[0].data.shape
    imageArray = np.zeros((numImages, row, col, 3)) #3 for diff, srch, tmpl triplet
    infoList = []

    for i in tqdm.tqdm(range(0, numImages, batchSize)):
        if i + batchSize > numImages:
            batchSize = numImages - i
            print(f'Corrected batchSize at image #{i}')
            
        batchIm, batchInfo = batchReadImages(filePaths, featuresDataFrame, i, batchSize)
        imageArray[i:i+batchSize] = batchIm
        # infoArray[i:i+batchSize] = batchInfo
        infoList.append(batchInfo)
        
    return imageArray, np.concatenate(infoList)

def normaliseImage(image, normType='standard', sigBound=3):
    """
    Function normalises single image data for training CNN, with options for different norm types
    
    Input:
        image: np.array, data for SINGLE image
        normType: type of normalisation to undertake
                    'standard' = simple normalisation from 0-->1, based on max and min
                    'gaussian' = normalisation around 𝜇=0, 𝜎=1 (primarily for diff)
                    'sigmaBounded' = for temp and srch images; normalises 𝜇 ± 3𝜎 interval onto 0->1, and leaves 
                                extreme values
        sigBound: relevant for sigmaBounded normType only; defaults to 3sigma normalisation
                                
    Returns:
        normalised image data in same format as input
    """
    
    if normType == 'standard':
        #scales images onto 0-->1
        normData = (image - image.min(keepdims=True)) / (image.max(keepdims=True) - image.min(keepdims=True))
        
        assert normData.min() == 0
        assert normData.max() == 1
    
    if normType == 'gaussian':  #typically what is used for diff images
        #normalises to Gaussian with mean = 0, std = 1
        normData = (image - image.mean(keepdims=True)) / (image.std(keepdims=True))
        
        assert np.isclose(normData.mean(), 0)
        assert np.isclose(normData.std(), 1)
    
    if normType == 'sigmaBounded': #typically used for tmpl and srch images to maintain extreme values
        #maps values inside sigma bounds onto 0-->1
        
        maxBound = image.mean(keepdims=True) + sigBound*image.std(keepdims=True)
        minBound = image.mean(keepdims=True) - sigBound*image.std(keepdims=True)
        
        maskInfo = np.ma.masked_where((image < minBound) | (image > maxBound), image)
        normData = np.where((image >= minBound) & (image <= maxBound), 
                            (image - maskInfo.min()) / (maskInfo.max() - maskInfo.min()), image)
        
        assert normData[maskInfo.mask == False].min() == 0
        assert normData[maskInfo.mask == False].max() == 1
        
    return normData

def randomCrop(image, outputSize = (25,25)):
    """
    Randomly crops image to desired output size, defaults to 25*25 pixels.
    
    Inputs:
        image: input image as np.array to be cropped. Can be any shape. 
        outputSize: array indicating shape of desired (2D) output, e.g. np.array([25,25]) OR (25,25) (either works). 
                    Can be any shape. 
                    
    Returns:
        Outputs cropped image in new size
    """
    
    #subtracting the output size makes sure that the mask doesn't go over the right/bottom edges
    topLeftX = np.random.choice(np.arange(image.shape[0] - outputSize[0]), 1)[0]
    topLeftY = np.random.choice(np.arange(image.shape[1] - outputSize[1]), 1)[0]
    
    if topLeftX+outputSize[0] > image.shape[0] or topLeftY+outputSize[1] > image.shape[1]:
        raise RuntimeError('randomImageCropping(): crop exceeds edges')
        
    outputImage = image[topLeftX:topLeftX+outputSize[0], topLeftY:topLeftY+outputSize[1]]
    return outputImage

def hstackImages(imageArray):
    """
    Horizontally stacks images from array with multiple channels.
    
    Input:
        imageArray: np.array of shape (a,b,c) where c = # of channels, (a,b) = image size
    
    Returns:
        np.array of shape (a, b*c), where images have been horizontally combined
    """
    
    shape = imageArray.shape
    #order = f reads indices fortran-like, so that it doesn't read depth-wise first
    
    return imageArray.reshape((shape[0], shape[1]*shape[2]), order='f')