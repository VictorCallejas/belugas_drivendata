import torch
from torchvision import transforms as T
from torchvision.io import read_image

import itertools

import pandas as pd 

from tqdm import tqdm

from sklearn.model_selection import train_test_split


IMG_SIZE = 224
ROOT_DIR = '../data/'
NORM_TRANSFORMS = torch.nn.Sequential(
    T.Resize([IMG_SIZE, IMG_SIZE]),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean = (0.4234, 0.4272, 0.4641),
                std  = (0.2037, 0.2027, 0.2142)),
)

VAL_SPLIT = 0.05


METADATA = pd.read_csv('../data/metadata.csv')


#TRAIN, VAL = train_test_split(METADATA, test_size=0.05, random_state=42)
#TRAIN, VAL = TRAIN.reset_index(), VAL.reset_index()
TRAIN, VAL = METADATA, METADATA

def getImages(metadata):
    
    IMAGES = {}
    for image_id, path in tqdm(zip(metadata.image_id, metadata.path), total=metadata.shape[0]):
        IMAGES[image_id] = NORM_TRANSFORMS(read_image(ROOT_DIR + path))
    
    return IMAGES

IMAGES = getImages(METADATA)

class PreTrain_BellugaDataset(torch.utils.data.Dataset):
    
    def __init__(self, metadata):
        
        self.metadata = metadata

    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, idx):
        return IMAGES[self.metadata.image_id[idx]]


class Eval_BellugaDataset(torch.utils.data.Dataset):
    def __init__(self, metadata):
        
        self.metadata = metadata
    
        # GROUND TRUTH
        gt = []
        for wid in self.metadata.whale_id: # query
            tmp = self.metadata[self.metadata.whale_id == wid].image_id.tolist() # get all images id
            gt.extend(list(itertools.permutations(tmp, 2)))
        self.gt = pd.DataFrame(gt,columns=['query_id','database_image_id'])
        self.gt = self.gt.set_index('query_id')
        
        # ALL QUERIES
        self.query_reference = list(itertools.permutations(self.metadata.image_id, 2))
            
    def getimage(self, image_id):
        return IMAGES[image_id]
        #path = self.image_path[image_id]
        #return self.transforms(read_image(path)

    def __len__(self):
        return len(self.query_reference)
    
    def __getitem__(self, idx):
        
        query_id = self.query_reference[idx][0]
        reference_id = self.query_reference[idx][1]
        
        query = self.getimage(query_id)
        reference = self.getimage(reference_id)
        
        return query, reference, query_id, reference_id
    
    
class Train_BellugaDataset(torch.utils.data.Dataset):
    def __init__(self, metadata):
        
        self.metadata = metadata
            
    def getimage(self, image_id):
        return IMAGES[image_id]
        #path = self.image_path[image_id]
        #return self.transforms(read_image(path))

    def __len__(self):
        return self.metadata.shape[0]
    
    def __getitem__(self, idx):
        
        anchor = self.getimage(self.metadata.image_id[idx])
        label = self.metadata.whale_id[idx]
        
        pos = self.getimage(self.metadata[self.metadata.whale_id == label].sample()['image_id'].values[0])
        neg = self.getimage(self.metadata[self.metadata.whale_id != label].sample()['image_id'].values[0])
            
        return anchor, pos, neg