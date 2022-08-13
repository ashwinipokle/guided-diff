from .mvtec_ad import MVTecAD
from torch.utils.data import ConcatDataset
from typing import Optional, Callable

ALL_CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                'transistor', 'wood', 'zipper'
                ]  

def get_all_mvtec_data(root,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None,
                 download=True,
                 pin_memory=False,
                 return_mask=False):
    
    all_datasets = []
    for category in ALL_CLASSES:
        dataset = MVTecAD(root=root, subset_name=category, train=train, transform=transform,
        target_transform=target_transform, mask_transform=mask_transform,
        download=download, pin_memory=pin_memory, return_mask=return_mask)
        all_datasets.append(dataset)

    combined = ConcatDataset(all_datasets)
    return combined



def get_mvtec_combined_data(root,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None,
                 download=True,
                 pin_memory=False,
                 return_mask=False):
    
    if train:
        trainset = get_all_mvtec_data(root=root, train=True, transform=transform, 
                                            target_transform=target_transform,
                                            mask_transform=mask_transform, 
                                            download=download, 
                                            pin_memory=pin_memory, 
                                            return_mask=return_mask)

        testset = get_all_mvtec_data(root=root, train=False, transform=transform, 
                                            target_transform=target_transform,
                                            mask_transform=mask_transform, 
                                            download=download, 
                                            pin_memory=pin_memory, 
                                            return_mask=return_mask)
    else:
        return get_all_mvtec_data(root=root, train=False, transform=transform, 
                                            target_transform=target_transform,
                                            mask_transform=mask_transform, 
                                            download=download, 
                                            pin_memory=pin_memory, 
                                            return_mask=return_mask
                                            )
    return combined
