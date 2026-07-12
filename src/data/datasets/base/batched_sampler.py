from typing import Callable, Optional
from collections import defaultdict

import numpy as np
from torch.utils.data import BatchSampler, DistributedSampler

from src.data.datasets.base.data_sampling_type import DataSamplingType
from src.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)


class CustomDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to support customized sampling
    dataset should be CatDataset: list[ResizedDataset]
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

        self.all_rank_sample_list = [] # we will set this in batchsampler
        self._aspect_ratio = None
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __len__(self):
        return len(self.all_rank_sample_list[self.rank])
    
    def __iter__(self):
        """
        Yields a sequence of (global_index, data_sampling_type, total_num_image, num_image, num_unsynced_video, num_synced_video, _aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        sample_list = self.all_rank_sample_list[self.rank]
        indices_iter = iter(sample_list)
        for indices in indices_iter:
            indices_new = indices + (self._aspect_ratio,)
            yield indices_new
    
    def update_parameters(self, _aspect_ratio):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            _aspect_ratio: The aspect ratio to set.
        """
        self._aspect_ratio = _aspect_ratio
        

class CustomBatchSampler(BatchSampler):
    def __init__(self, sampler, epoch=0, seed=42,
                 max_image_per_gpu=24, _aspect_ratio_range=None):
        """
        Initializes the batch sampler
        This sampler is designed to let each dataset sample its own samples according to its own sampling rule, 
        rather than use a global sampling rule. All batches wiil be allocated in this class and assign to
        CustomDistributedSampler in set_epoch. Batches for all ranks in one step will have the same batch_size, total_num_image,
        _aspect_ratio to support distributed training. Samples within one batch will have the same sampling logic (the same num_time)

        Args:
            sampler: instance of CustomDistributedSampler
            epoch: int
            seed: int
            max_image_per_gpu: int
            _aspect_ratio_range: tuple
        """

        super().__init__(sampler, batch_size=1, drop_last=False) # dummy

        self.sampler = sampler
        self.rng = np.random.default_rng(seed=seed)
        self._aspect_ratio_range = _aspect_ratio_range

        self.max_image_per_gpu = max_image_per_gpu
        self.world_size = self.sampler.num_replicas
        
        self.all_rank_sample_list = []
        self.other_params = []
        self.set_epoch(epoch+seed)
    

    def _generate_sample_for_dataset(self, dataset_length, data_sampling_info):
        """
        generate sample_list for dataset
        Args:
            dataset_length: int, dataset length
            data_sampling_info: dict, data_sampling_type_info
                data_sampling_type: dict
                    sampling_ratio: float
                    max_total_image: int
                    max_unsynced_video: int
                    max_synced_video: int
                    max_oribit_frame: int
                    hybrid_unsynced_video_prob: float
        Returns:
            index_list: list, list of index
            sample_list: list, list of sample_info, length might be slightly lower than dataset_length
                sample_info: dict, batch should contains samples of exactly the same sample_info
                    data_sampling_type: DataSamplingType
                    total_num_image: int
                    num_image: [optional] int
                    num_synced_video: [optional] int
                    num_unsynced_video: [optional] int
        """
        sample_list = []
        # normalize data_sampling_ratio if need
        total_sampling_ratio = sum([type_info["sampling_ratio"] for type_info in data_sampling_info.values()])
        if total_sampling_ratio != 1.0:
            for type_info in data_sampling_info.values():
                type_info["sampling_ratio"] /= total_sampling_ratio
        
        for data_sampling_type, type_info in data_sampling_info.items():
            sampling_ratio = type_info["sampling_ratio"]
            if sampling_ratio == 0.0:
                continue
            min_total_image = type_info["min_total_image"]
            max_total_image = type_info["max_total_image"]
            min_static_image = type_info.get("min_static_image", 0)
            min_unsynced_video = type_info.get("min_unsynced_video", 0)
            max_unsynced_video = type_info.get("max_unsynced_video", 0)
            min_synced_video = type_info.get("min_synced_video", 0)
            max_synced_video = type_info.get("max_synced_video", 0)
            max_oribit_frame = type_info.get("max_oribit_frame", 0)
            hybrid_unsynced_video_prob = type_info.get("hybrid_unsynced_video_prob", 0.5)

            num_data = int(sampling_ratio * dataset_length)
            num_remaining_data = num_data
            MAX_RETRY = 5
            num_retry = 0
            while num_retry < MAX_RETRY:
                num_retry += 1
                while num_remaining_data > 0:
                    # sample batch info
                    cur_sample_info = {
                        "data_sampling_type": data_sampling_type,
                        "total_num_image": 1,
                        "num_image": 0,
                        "num_synced_video": 0,
                        "num_unsynced_video": 0,
                    }

                    if data_sampling_type == DataSamplingType.StaticImage:
                        total_num_image = self.rng.integers(min_total_image, max_total_image+1) # min_total_image >=1
                        cur_sample_info["total_num_image"] = total_num_image
                        cur_sample_info["num_image"] = total_num_image
                    elif data_sampling_type == DataSamplingType.UnsyncedDynamicVideo:
                        # note: num_unsynced_video >=2, allow different num_video_frame (>=1) for each unsynced video
                        total_num_image = self.rng.integers(min_total_image, max_total_image+1) # min_total_image >=2
                        num_unsynced_video = self.rng.integers(min_unsynced_video, min(total_num_image, max_unsynced_video)+1) # min_unsynced_video >=2
                        cur_sample_info["total_num_image"] = total_num_image
                        cur_sample_info["num_unsynced_video"] = num_unsynced_video
                    elif data_sampling_type == DataSamplingType.SyncedDynamicVideo:
                        # note: num_synced_video >=1, at least 2 frames per video
                        total_num_image = self.rng.integers(min_total_image, max_total_image+1) # min_total_image >=4
                        num_synced_video = self.rng.integers(min_synced_video, min(max_synced_video, total_num_image//2)+1) # min_synced_video >=1
                        total_num_image = total_num_image // num_synced_video * num_synced_video
                        cur_sample_info["total_num_image"] = total_num_image
                        cur_sample_info["num_synced_video"] = num_synced_video
                    elif data_sampling_type == DataSamplingType.HybridDynamicVideo:
                        # note: at least 2 image for oribit video(only 1 video), 
                        # for unsynced_video, num_unsynced_video >=2, at least one frame for each video, total_num_image >= 4
                        # for synced_video, num_synced_video >=1, at least one frame for each video, total_num_image >= 3
                        if self.rng.random() < hybrid_unsynced_video_prob:
                            total_num_image = self.rng.integers(min_total_image, max_total_image+1) # min_total_image >=4
                            num_static_image = self.rng.integers(min_static_image, min(total_num_image - 2, max_oribit_frame)+1) # min_static_image >=2
                            num_dynamic_image = total_num_image - num_static_image
                            num_unsynced_video = self.rng.integers(min_unsynced_video, min(max_unsynced_video, num_dynamic_image)+1)
                            cur_sample_info["num_unsynced_video"] = num_unsynced_video
                        else:
                            total_num_image = self.rng.integers(min_total_image, max_total_image+1) # min_total_image >=3
                            num_static_image = self.rng.integers(min_static_image, min(total_num_image - 1, max_oribit_frame)+1) # min_static_image >=2
                            num_dynamic_image = total_num_image - num_static_image
                            num_synced_video = self.rng.integers(min_synced_video, min(max_synced_video, num_dynamic_image)+1) # min_synced_video >=1
                            num_dynamic_image = num_dynamic_image // num_synced_video * num_synced_video
                            cur_sample_info["num_synced_video"] = num_synced_video
                       
                        total_num_image = num_static_image + num_dynamic_image
                        cur_sample_info["total_num_image"] = total_num_image
                        cur_sample_info["num_image"] = num_static_image
                    

                    cur_batch_size = self.max_image_per_gpu // total_num_image
                    # if not enough data, break
                    if num_remaining_data < cur_batch_size:
                        break
                    
                    # if possible, we should assign the batch for all ranks
                    for _ in range(self.world_size):
                        num_remaining_data -= cur_batch_size
                        if num_remaining_data < 0:
                            break
                        sample_list.extend([cur_sample_info for _ in range(cur_batch_size)])
                                
        index_list = self.rng.choice(dataset_length, size=len(sample_list), replace=False).tolist()

        return index_list, sample_list

    def _merge_data_sample_lists(self, all_index_list, all_sample_list, dataset_cum_sizes):
        """
        merge data sample lists from all dataset
        Args:
            all_index_list: list of list, list of index list for each dataset
            all_sample_list: list of list, list of sample list for each dataset
            dataset_cum_sizes: list, list of dataset cumulative sizes
        Returns:
            batched_dict: dict, key is total_num_image, val is list of batch
        """
        
        # 临时存储分组数据
        temp_dict = defaultdict(lambda: defaultdict(list))

        # 先分好 total_num_image -> sub_key -> list of val
        for dataset_idx, (index_list, sample_list) in enumerate(zip(all_index_list, all_sample_list)):
            offset = dataset_cum_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
            global_index_list = [index + offset for index in index_list]
            for global_index, sample in zip(global_index_list, sample_list):

                total_num_image = sample["total_num_image"]  # 第一层 key， for distributed sampling
                sub_key = (
                    sample["data_sampling_type"],
                    sample["num_image"],
                    sample["num_unsynced_video"],
                    sample["num_synced_video"]
                )  # 第二层 key， for batching

                # val
                val = (
                    global_index,
                    sample["data_sampling_type"],
                    total_num_image,
                    sample["num_image"],
                    sample["num_unsynced_video"],
                    sample["num_synced_video"]
                )

                temp_dict[total_num_image][sub_key].append(val)

        # 打乱组成batch的sample顺序
        [self.rng.shuffle(v) for g in temp_dict.values() for v in g.values()]

        # 开始按 batch 组装
        batched_dict = {}

        for total_num_image, sub_groups in temp_dict.items():
            batch_size = self.max_image_per_gpu // total_num_image
            batched_list = []  # 当前 total_num_image 下的所有 batch

            for sub_key, values in sub_groups.items():
                # 按 batch_size 切块
                # TODO: 如果我们需要每个gpu上有不同的aspect_ratio，可以在这一步加
                for i in range(0, len(values) // batch_size):
                    batch = values[i * batch_size:(i + 1) * batch_size]
                    batched_list.append(batch)

            batched_dict[total_num_image] = batched_list
        
        # 打乱分给各个rank的batch顺序
        [self.rng.shuffle(v) for v in batched_dict.values()]
                
        return batched_dict


    def _distribute_batches(self, batched_dict):
        """
        简化分配方法（不存 total_num_image）：
        1. 按 total_num_image 切成 world_size 份并分配
        2. 全部分配完后统一打乱索引
        3. 统计 batch_size_per_step（直接用 len(batch)）
        4. 展平成 sample_list
        """
        
        # 初始化 rank -> batch 列表
        rank_batches = [[] for _ in range(self.world_size)]
        
        # step1: 按 total_num_image 切分
        for total_num_image, batches in batched_dict.items():
            num_batches = len(batches)
            per_rank = num_batches // self.world_size
            usable_batches = per_rank * self.world_size
            
            if usable_batches == 0:
                continue  # 不够分就跳过
            
            batches = batches[:usable_batches]
            
            # 切分成 world_size 份
            split_batches = [batches[i * per_rank:(i + 1) * per_rank] for i in range(self.world_size)]
            
            # 直接扩展 batch 列表
            for r in range(self.world_size):
                rank_batches[r].extend(split_batches[r])
        
        # step2: 生成统一的打乱索引
        num_steps = len(rank_batches[0])
        perm = self.rng.permutation(num_steps)
        
        # step3: 按相同顺序打乱所有 rank 的 batches
        for r in range(self.world_size):
            rank_batches[r] = [rank_batches[r][i] for i in perm]
        
        # step4: 统计 batch_size_per_step（直接统计 batch 的长度）
        batch_size_per_step = [len(batch) for batch in rank_batches[0]]
        
        # step5: 展平成 sample_list
        all_rank_sample_list = []
        for r in range(self.world_size):
            samples = []
            for batch in rank_batches[r]:
                samples.extend(batch)  # batch 是 list/tuple of samples
            all_rank_sample_list.append(samples)
        
        return batch_size_per_step, all_rank_sample_list
        
        
    def build_all_rank_sample_list(self):
        """
        build all rank sample list
        """
        cat_dataset = self.sampler.dataset
        dataset_cum_sizes = cat_dataset._cum_sizes

        all_index_list = []
        all_sample_list = []
        # sampling each dataset
        for single_dataset in cat_dataset.datasets:
            index_list, sample_list = self._generate_sample_for_dataset(len(single_dataset), single_dataset.data_sampling_info)
            all_index_list.append(index_list)
            all_sample_list.append(sample_list)
        
        batched_dict = self._merge_data_sample_lists(all_index_list, all_sample_list, dataset_cum_sizes)
        batch_size_per_step, all_rank_sample_list = self._distribute_batches(batched_dict)

        # sample _aspect_ratio
        _aspect_ratio_per_step = [tuple(self.rng.uniform(self._aspect_ratio_range[0],
                                                    self._aspect_ratio_range[1], size=2).tolist()) for _ in range(len(batch_size_per_step))]
        
        self.other_params = [{"batch_size":batch_size_, "_aspect_ratio":_aspect_ratio_} for (batch_size_, _aspect_ratio_) in zip(batch_size_per_step, _aspect_ratio_per_step)]
        self.all_rank_sample_list = all_rank_sample_list
        self.sampler.all_rank_sample_list = all_rank_sample_list
    
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng = np.random.default_rng(seed=epoch + 777)
        self.build_all_rank_sample_list()
        total_num_sample = sum(sample["batch_size"] for sample in self.other_params)
        log.info(f"BatchSampler epoch: {epoch} total samples: {total_num_sample}")
    
    def __iter__(self):
        """
        按预生成的 self.batch_size_per 逐批消费底层 sampler。
        每个 batch 里样本共享同一 _aspect_ratio / total_num_image / etc
        """
        sampler_iterator = iter(self.sampler)
        for other_params in self.other_params:
            # 把动态参数同步给底层 sampler（供 dataset 使用）
            self.sampler.update_parameters(
                _aspect_ratio=other_params["_aspect_ratio"],
            )
            current_batch = []
            for _ in range(other_params["batch_size"]):
                try:
                    item = next(sampler_iterator)  # (global_index, data_sampling_type, num_image, num_unynsced_video, num_synced_video, _aspect_ratio)
                    current_batch.append(item)
                except StopIteration:
                    break
            if not current_batch:
                break
            yield current_batch
    
    def __len__(self):
        return len(self.other_params)
