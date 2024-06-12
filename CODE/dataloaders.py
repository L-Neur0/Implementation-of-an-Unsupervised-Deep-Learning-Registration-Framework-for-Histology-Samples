import os

import numpy as np
import random
import SimpleITK as sitk

import torch
import torch.utils as utils

def collate_to_list_unsupervised(batch):
    sources = [item[0].view(item[0].size(0), item[0].size(1)) for item in batch]
    targets = [item[1].view(item[1].size(0), item[1].size(1)) for item in batch]
    return sources, targets

def collate_to_list_segmentation(batch):
    sources = [item[0].view(item[0].size(0), item[0].size(1)) for item in batch]
    targets = [item[1].view(item[1].size(0), item[1].size(1)) for item in batch]
    source_masks = [item[2].view(item[2].size(0), item[2].size(1)) for item in batch]
    target_masks = [item[3].view(item[3].size(0), item[3].size(1)) for item in batch]
    return sources, targets, source_masks, target_masks

class UnsupervisedLoader(utils.data.Dataset):
    def __init__(self, data_path, transforms=None, randomly_swap=False):
        self.data_path = data_path
        self.all_ids = os.listdir(self.data_path)
        self.transforms = transforms
        self.randomly_swap = randomly_swap
        
        
    #restituisce la lunghezza totale del dataset, che è il numero di campioni presenti nella directory dei dati.
    def __len__(self): 
        return len(self.all_ids)

    #prende un indice idx come input e carica il campione corrispondente a quell'indice. 
    # Qui è dove avviene il caricamento effettivo dei dati:
    def __getitem__(self, idx): 
        #Estrae l'identificatore del caso corrispondente all'indice fornito.
        case_id = self.all_ids[idx]
        #Costruisce i percorsi dei file per i file sorgente e target utilizzando l'identificatore del caso.
        source_path = os.path.join(self.data_path, str(case_id), "source.mha")
        target_path = os.path.join(self.data_path, str(case_id), "target.mha")
        #Carica i dati dei file sorgente e target utilizzando SimpleITK.
        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))

        if self.transforms is not None:
            source, target = self.transforms(source, target)

        if self.randomly_swap:
            if random.random() > 0.5:
                pass
            else:
                source, target = target, source

        source_tensor, target_tensor = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        return source_tensor, target_tensor

class SegmentationLoader(utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_ids = os.listdir(self.data_path)

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        case_id = self.all_ids[idx]
        source_path = os.path.join(self.data_path, str(case_id), "source.mha")
        target_path = os.path.join(self.data_path, str(case_id), "target.mha")
        source_mask_path = os.path.join(self.data_path, str(case_id), "source_mask.mha")
        target_mask_path = os.path.join(self.data_path, str(case_id), "target_mask.mha")
        source = sitk.GetArrayFromImage(sitk.ReadImage(source_path))
        target = sitk.GetArrayFromImage(sitk.ReadImage(target_path))
        source_mask = sitk.GetArrayFromImage(sitk.ReadImage(source_mask_path))
        target_mask = sitk.GetArrayFromImage(sitk.ReadImage(target_mask_path))

        source_tensor, target_tensor = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
        source_mask_tensor, target_mask_tensor = torch.from_numpy(source_mask.astype(np.float32)), torch.from_numpy(target_mask.astype(np.float32))
        return source_tensor, target_tensor, source_mask_tensor, target_mask_tensor