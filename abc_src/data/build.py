
import torch
from torch.utils.data import DataLoader
import numpy as np

from data.datasets.kitti import KITTIDataset
from data.transforms import build_transforms
from structures.image_list import to_image_list
from data.sampler import TrainingSampler, InferenceSampler

from utils.init_fn import seed_all_rng
from utils.comm import get_world_size


def build_dataset(cfg, split, transforms, is_train: bool, is_visualize=False):
    return KITTIDataset(cfg, transforms, is_train, split, is_visualize)


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return dict(images=images, targets=targets, img_ids=img_ids)


def collate_fn(batch):
    batch = list(zip(*batch))
    return {"images": torch.stack(batch[0], 0),
            "targets": batch[1], "img_ids": batch[2]}


def build_dataloader(cfg, split, is_train=True, shuffle=False, is_visualize=False, bsize=None):
    dataset = build_dataset(cfg, split, build_transforms(), is_train, is_visualize)
    if not bsize:
        bsize = cfg.SOLVER.BATCH_SIZE if is_train else 1
    num_workers = cfg.DATALOADER.NUM_WORKERS   # 2
    collator = BatchCollator()
    return DataLoader(
        dataset,
        batch_size=bsize,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,  # False, 若显存不够
        drop_last=True,
        worker_init_fn=worker_init_reset_seed,
    )


def make_data_loader(cfg, split, is_train=True, is_visualize=False, bsize=None):
    dataset = build_dataset(cfg, split, build_transforms(), is_train, is_visualize)
    if bsize is None:
        bsize = cfg.SOLVER.BATCH_SIZE if is_train else 1
    num_workers = cfg.DATALOADER.NUM_WORKERS  # 2
    collator = BatchCollator()

    num_gpus = get_world_size()
    images_per_gpu = bsize // num_gpus

    sampler = TrainingSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=True
    )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=True,
        worker_init_fn=worker_init_reset_seed,
    )


def build_test_loader(cfg, split, is_train=False, is_visualize=True):
    dataset = build_dataset(cfg, split, build_transforms(), is_train, is_visualize)
    num_workers = 2  # 2, 0
    collator = BatchCollator()

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False
    )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=True,
    )


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


if __name__ == "__main__":
    from config import cfg
    # tr_loader = make_data_loader(cfg, split='train', is_train=True)
    # for batch, iteration in zip(tr_loader, range(0, 46400)):
    #     print(iteration)

    va_loader = build_test_loader(cfg, split='val', is_train=False, is_visualize=False)
    # for batch, idx in zip(va_loader, range(0, 100)):
    for idx, batch in enumerate(va_loader):
        print(idx)

