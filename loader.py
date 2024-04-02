import numpy as np
import pandas as pd

import pathlib
import string
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch

from sklearn.model_selection import train_test_split

from base import BaseDataset
from models import Segmentation

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def sample_from_letter(fnm_list, n_items, case = None):
    #case: ('lower','upper')
    df = pd.DataFrame()
    df['fname'] = [fnm.split('.')[0] for fnm in fnm_list]
    spl = df.fname.str.split('_')
    df['letter'] = spl.apply(lambda x: x[0])
    df['case'] = spl.apply(lambda x: x[-1])
    df['class'] = df['letter']+df['case']
    df['class'] = pd.Categorical(df['class'])
    df['label'] = df['class'].cat.codes    

    n_classes = df['class'].nunique()
    
    if case is not None:
        df = df[df.case == case]
    samples = df.groupby('label').apply(lambda x: x.sample(n_items))
    fnm_labels = samples[['fname','label']].set_index('fname').label.to_dict()
    return n_classes, fnm_labels

class RankingDataset(BaseDataset):
    @staticmethod
    def num_classes():
        return self.num_classes

    def __init__(
        self,
        root_dir,
        fnm_labels,
        num_classes,
        _center_and_scale=True,
        random_rotate=False,
    ):
        """
        Args:
            _center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        # path = pathlib.Path(root_dir)
        self.random_rotate = random_rotate
        self.num_classes = num_classes
        
        self.lbs = fnm_labels

        file_paths = [pathlib.Path(root_dir+fnm+'.bin') for fnm in fnm_labels.keys()]
        print(file_paths[0], file_paths[0].exists())
        self.load_graphs(file_paths, _center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        sample = super().load_one_graph(file_path)
        # Additionally get the label from the filename and store it in the sample dict

        sample["label"] = torch.tensor([self.lbs[str(file_path.stem)]]).long()
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        return collated
    
    
def get_loaders(model, batch_size=128, random_rotate=False, num_workers=0, num_items=100):
    args = AttrDict({})
    args.batch_size = batch_size
    args.num_workers = num_workers
    fnm_list = os.listdir('D:/NIR/SolidLetters/graph_with_eattr')
    N_ITEMS_PER_CLASS = num_items
# creating loaders for SolidLetters dataset quering
    test_loaders = []
    for case in ('lower', 'upper'):
        ncl, fnm_labels = sample_from_letter(fnm_list, N_ITEMS_PER_CLASS, case)
        dset = RankingDataset('D:/NIR/SolidLetters/graph_with_eattr/', 
                               fnm_labels, 
                               ncl)
        test_loaders.append(dset.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers))
    return test_loaders
