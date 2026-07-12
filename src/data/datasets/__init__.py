import torch

from torch.utils.data.distributed import DistributedSampler
from src.data.datasets.base.easy_dataset import CatDataset, ResizedDataset
from src.data.datasets.base.batched_sampler import CustomBatchSampler, CustomDistributedSampler
from .waymo import WaymoDataset
from .waymo_test import WaymoTestDataset
from .ue import UEDataset
from .ue_test import UETestDataset
from .omnigame import OmniGameDataset
from .omnigame_test import OmniGameTestDataset
from .dynamic_replica import DynamicReplicaDataset
from .dynamic_replica_test import DynamicReplicaTestDataset
from .stereo4d import Stereo4dDataset
from .stereo4d_test import Stereo4dTestDataset
from .spring import SpringDataset
from .spring_test import SpringTestDataset
from .hoi4d import HOI4dDataset
from .hoi4d_test import HOI4dTestDataset
from .point_odyssey import PointOdysseyDataset
from .point_odyssey_test import PointOdysseyTestDataset
from .dl3dv import DL3DVDataset
from .dl3dv_test import DL3DVTestDataset
from .droid import DroidDataset
from .droid_test import DroidTestDataset

DATASET_REGISTRY = {
    "waymo": WaymoDataset,
    "waymo_test": WaymoTestDataset,
    "ue": UEDataset,
    "ue_test": UETestDataset,
    "omnigame": OmniGameDataset,
    "omnigame_test": OmniGameTestDataset,
    "dynamic_replica": DynamicReplicaDataset,
    "dynamic_replica_test": DynamicReplicaTestDataset,
    "stereo4d": Stereo4dDataset,
    "stereo4d_test": Stereo4dTestDataset,
    "spring": SpringDataset,
    "spring_test": SpringTestDataset,
    "hoi4d": HOI4dDataset,
    "hoi4d_test": HOI4dTestDataset,
    "point_odyssey": PointOdysseyDataset,
    "point_odyssey_test": PointOdysseyTestDataset,
    "dl3dv": DL3DVDataset,
    "dl3dv_test": DL3DVTestDataset,
    "droid": DroidDataset,
    "droid_test": DroidTestDataset,
}


## dist
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


## main func
def build_dataset(dataset_cfgs, stage="train"):
    datasets = []

    for cfg in dataset_cfgs:
        cls_name = cfg['class']

        if cls_name not in DATASET_REGISTRY:
            raise ValueError(f"Dataset class {cls_name} not found in registry.")

        dataset_cls = DATASET_REGISTRY[cls_name]
        base_dataset = dataset_cls(**cfg.get('args', {}))

        if stage == "train" and 'size' in cfg:
            base_dataset = ResizedDataset(cfg['size'], base_dataset)

        datasets.append(base_dataset)

    return CatDataset(datasets) if stage=="train" else datasets


def build_custom_sampler(dataset, rank, num_replicas, images_per_gpu, aspect_ratio_range):
    sampler = CustomDistributedSampler(
        dataset=dataset,
        rank=rank,
        num_replicas=num_replicas,
    )

    return CustomBatchSampler(sampler, max_image_per_gpu=images_per_gpu, \
            _aspect_ratio_range=aspect_ratio_range)


def get_data_loader(
    dataset,
    images_per_gpu=None,
    num_workers=8,
    shuffle=False,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False, 
    multiprocessing_context=None,
    aspect_ratio_range=None,
    stage="train",
):

    world_size = get_world_size()
    rank = get_rank()

    if stage == "train":
        batch_sampler = build_custom_sampler(dataset, rank, world_size, images_per_gpu, \
                                            aspect_ratio_range)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

    else:

        assert shuffle==False, "shuffle must be False in eval mode"
        sampler = DistributedSampler(dataset, shuffle=False)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

    return data_loader