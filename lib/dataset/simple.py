"""Simple datasets and dataset utilities."""
import os
import torch
from PIL import Image
import numpy as np
from lib.utils import construct_class_by_name, cat_dict


def get_exemplar_data(ds, n_sample=4):
    """Get exemplar data from dataset."""
    #indices = torch.randint(0, len(ds), (n_sample,))
    indices = torch.arange(0, n_sample)
    if isinstance(ds, UnionDataset):
        datas = [ds[i] for i in indices]
    elif isinstance(ds, ConcatDataset):
        datas = [ds.get_union_item(i) for i in indices]
    else:
        ds.ds_keys = ['default']
        datas = [{'default': ds[i]['ref_camera']} for i in indices]
    return datas


def process_data(processor_fn, data):
    """Process data with a given processor function.
    For each dataset item, process each camera component, concatenate camera components
    and then concatenate dataset items.
    Args:
        data: list of dict, each dict is of two level: first level is dataset key 
            (specified in config), second level is camera key (ref_camera / tar_camera), each camera key stores a list of dict.
    Returns:
        dict: processed data.
    """
    ref_datas = []
    for ds_key in data[0].keys():
        batch = {}
        cam_keys = [key for key in data[0][ds_key].keys() if 'camera' in key]
        for cam_key in cam_keys:
            # first stack: image to tensor batch
            # (N_BATCH, N_VIEWS, ...)
            batch[cam_key] = cat_dict([d[ds_key][cam_key] for d in data])
        ref_datas.append(processor_fn(batch)[0])
    # second stack: concat different dataset
    dic = cat_dict(ref_datas, is_stack=False)
    return dic


class ConcatDataset(torch.utils.data.Dataset):
    """Concatenate items of multiple datasets."""
    def __init__(self, ds_list, ds_keys):
        self.ds_list = [construct_class_by_name(**dic) for dic in ds_list]
        self.ds_keys = ds_keys
        self.cumulative_lens = np.cumsum([len(ds) for ds in self.ds_list])

    def __len__(self):
        return self.cumulative_lens[-1]

    def __repr__(self):
        strs = [f'{ds_name}={len(ds)}'
                for ds_name, ds in zip(self.ds_keys, self.ds_list)]
        return ' '.join(strs)

    def get_union_item(self, idx):
        return {self.ds_keys[i]: ds[idx % len(ds)]
                for i, ds in enumerate(self.ds_list)}

    def __getitem__(self, idx):
        for i, cum_len in enumerate(self.cumulative_lens):
            if idx < cum_len:
                ds_in_ind = idx - self.cumulative_lens[i - 1] if i > 0 else idx
                return self.ds_keys[i], self.ds_list[i][ds_in_ind]


class UnionDataset(torch.utils.data.Dataset):
    """Return items of multiple datasets in a dictionary."""
    def __init__(self, ds_list, ds_keys):
        self.ds_list = [construct_class_by_name(**dic) for dic in ds_list]
        self.ds_keys = ds_keys

    def __len__(self):
        return max([len(ds) for ds in self.ds_list])

    def __getitem__(self, idx):
        return {self.ds_keys[i]: ds[idx % len(ds)]
                for i, ds in enumerate(self.ds_list)}


class SimpleDataset(torch.utils.data.Dataset):
    """
    Image-only datasets.
    """
    def __init__(self, data_path, size=None, transform=None):
        self.size = size
        self.data_path = data_path
        self.transform = transform

        image_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    file_path = os.path.join(root, file)
                    #if 'masks' in file_path:
                    #    continue
                    image_files.append(file_path)
        image_files.sort()
        self.files = image_files

    def __getitem__(self, idx):
        fpath = self.files[idx]
        with open(fpath, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.size:
                img = img.resize(self.size, Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'image_path': fpath
        }

    def __len__(self):
        return len(self.files)
