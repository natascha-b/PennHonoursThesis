from data_utils import *
import h5py
import pickle

#change filepaths as necessary
featureinfoPath = 'autoscanFiles/autoscan_features.3.csv'
stampsDirectory = 'stampDirectory'

outputImageFile = 'imageArray.h5' #must be .h5 file
outputLabelFile = 'labels.pk' #must be .pk file

#reading in images
featuredf = pd.read_csv(featureinfoPath, skiprows=6)
stampPaths = getFilepaths(stampsDirectory)
imgs, info = openBatched(stampPaths, featuredf, batchSize=25)

#imgs are in the order: diff, tmpl, srch
for i, triplet in enumerate(tqdm.tqdm(imgs)):
    imgs[i,:,:,0] = normaliseImage(imgs[i,:,:,0], normType='gaussian')
    imgs[i,:,:,1] = normaliseImage(imgs[i,:,:,1], normType='sigmaBounded', sigBound=3)
    imgs[i,:,:,2] = normaliseImage(imgs[i,:,:,2], normType='sigmaBounded', sigBound=3)

with h5py.File(outputImageFile, 'w') as f:
    f.create_dataset('stamps', data=imgs)

with open(outputLabelFile, 'wb') as f:
    pickle.dump(info, f)