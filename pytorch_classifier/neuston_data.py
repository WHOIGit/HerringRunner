"""this module handles the parsing of data directories"""

# built in imports
import os, sys
import random

# 3rd party imports
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset, IterableDataset
from torch import tensor
import pandas as pd

# project imports
import ifcb
from ifcb.data.adc import SCHEMA_VERSION_1
from ifcb.data.stitching import InfilledImages

cats=['A','B','C','D','E']
cat2count = {1:(-1,0,'A'),
             2:(1,4,'B'),
             3:(5,10,'C'),
             4:(11,20,'D'),
             5:(21,40,'E')}
def count2cat(x):
    if x<=cat2count[1][1]: return cat2count[1][2]
    elif x<=cat2count[2][1]: return cat2count[2][2]
    elif x<=cat2count[3][1]: return cat2count[3][2]
    elif x<=cat2count[4][1]: return cat2count[4][2]
    elif x>=cat2count[5][0]: return cat2count[5][2]
    else: raise ValueError(x)


## TRAINING ##

class HerringrunnerDataset(Dataset):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    Example setup:     dataloader = torch.utils.DataLoader(ImageFolderWithPaths("path/to/your/perclass/image/folders"))
    Example usage:     for inputs,labels,paths in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, image_paths, resize=244, mode=['count','cat'][1]):
        images = [img for img in image_paths if img.endswith(datasets.folder.IMG_EXTENSIONS)]
        labels = [os.path.splitext(img)[0].replace('/images/','/labels/')+'.txt' for img in images]
        both = [(i,l) for i,l in zip(images,labels) if os.path.isfile(i) and os.path.isfile(l)]
        self.image_paths,self.label_paths = zip(*both)
        self.image_counts = []
        for label_file in self.label_paths:
            with open(label_file) as f:
                count = len(f.readlines())
                self.image_counts.append(count)
        self.image_cat_labels = [count2cat(c) for c in self.image_counts]
        self.image_cats = [cats.index(c) for c in self.image_cat_labels]
        self.classes = cats
        self.mode = mode
        self.targets = self.image_counts if mode=='count' else self.image_cats
        
        # use 299x299 for inception_v3, all other models use 244x244
        self.transform = transforms.Compose([transforms.Resize([resize, resize]),
                                             transforms.ToTensor()])

        if len(self.image_paths) < len(image_paths):
            print('{} missing files were ommited'.format(len(image_paths)-len(self.image_paths)))
        if len(self.image_paths) == 0:
            raise RuntimeError('No images Loaded!!')

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = datasets.folder.default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        return image, tensor([target]), path

    def __len__(self):
        return len(self.image_paths)
        
    @property
    def images_perclass(self):
        ipc = {c: [] for c in self.classes}
        for img, trg in zip(self.image_paths, self.targets):
            ipc[self.classes[trg]].append(img)
        return ipc

    @property
    def count_perclass(self):
        cpc = [0 for c in self.classes] # initialize list at 0-counts
        for class_idx in self.image_cats:
            cpc[class_idx] += 1
        return cpc


class HerringRUNnerDataset(Dataset):
    """
    Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder
    Example setup:     dataloader = torch.utils.DataLoader(ImageFolderWithPaths("path/to/your/perclass/image/folders"))
    Example usage:     for inputs,labels,paths in my_dataloader: ....
    instead of:        for inputs,labels in my_dataloader: ....
    adapted from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, image_paths, resize=244, input_src=None):
        self.input_src = input_src
        self.image_paths = [img for img in image_paths if img.endswith(datasets.folder.IMG_EXTENSIONS) and os.path.isfile(img)]

        # use 299x299 for inception_v3, all other models use 244x244
        self.transform = transforms.Compose([transforms.Resize([resize, resize]),
                                             transforms.ToTensor()])

        if len(self.image_paths) < len(image_paths):
            print('{} missing files were ommited'.format(len(image_paths)-len(self.image_paths)))
        if len(self.image_paths) == 0:
            raise RuntimeError('No images Loaded!!')

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = datasets.folder.default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, path

    def __len__(self):
        return len(self.image_paths)




def get_trainval_datasets(args):
    print('Initializing Data...')
    
    resize = 299 if args.MODEL == 'inception_v3' else 224
    
    with open(args.TRAINING) as f1, open(args.VALIDATION) as f2:
        training_list = [line.strip() for line in f1.readlines()]
        validation_list = [line.strip() for line in f2.readlines()]
        
    mode = 'count' if args.counts_mode else 'cat'
    
    training_dataset = HerringrunnerDataset(training_list, resize, mode)
    validation_dataset = HerringrunnerDataset(validation_list, resize, mode)

    return training_dataset, validation_dataset


def parse_imgnorm(img_norm_arg):
    mean = img_norm_arg[0]
    mean = [float(m) for m in mean.split(',')]
    if len(mean) == 1: mean = 3*mean
    std = img_norm_arg[1]
    std = [float(s) for s in std.split(',')]
    if len(std) == 1: std = 3*std
    assert len(mean) == len(std) == 3, '--img-norm invalid: {}'.format(img_norm_arg)
    return mean,std



