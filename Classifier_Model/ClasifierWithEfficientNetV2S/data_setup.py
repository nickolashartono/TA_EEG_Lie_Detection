import os

from dataset import PickleLatentDatasetLoader
from dataset_img import ImgDatasetLoader
from dataset_img_dryad import ImgDatasetLoader as DryadImgDatasetLoader
from dataset_dryad import PickleLatentDatasetLoader as DryadDatasetLoader
from torch.utils.data import DataLoader
import pandas as pd

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    pkl_dir: str,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

    datasets_train = PickleLatentDatasetLoader(
        annotations_file=train_dir,
        dir=pkl_dir
    )
    datasets_valid = PickleLatentDatasetLoader(
        annotations_file=valid_dir,
        dir=pkl_dir
    )
    datasets_test = PickleLatentDatasetLoader(
        annotations_file=test_dir,
        dir=pkl_dir
    )

    # datasets_train = ImgDatasetLoader(
    #     annotations_file=train_dir,
    #     img_dir=pkl_dir
    # )
    # datasets_valid = ImgDatasetLoader(
    #     annotations_file=valid_dir,
    #     img_dir=pkl_dir
    # )
    # datasets_test = ImgDatasetLoader(
    #     annotations_file=test_dir,
    #     img_dir=pkl_dir
    # )

    # datasets_train = DryadImgDatasetLoader(
    #     annotations_file=train_dir,
    #     img_dir=pkl_dir
    # )
    # datasets_valid = DryadImgDatasetLoader(
    #     annotations_file=valid_dir,
    #     img_dir=pkl_dir
    # )
    # datasets_test = DryadImgDatasetLoader(
    #     annotations_file=test_dir,
    #     img_dir=pkl_dir
    # )

    # datasets_train = DryadDatasetLoader(
    #     annotations_file=train_dir,
    #     dir=pkl_dir
    # )
    # datasets_valid = DryadDatasetLoader(
    #     annotations_file=valid_dir,
    #     dir=pkl_dir
    # )
    # datasets_test = DryadDatasetLoader(
    #     annotations_file=test_dir,
    #     dir=pkl_dir
    # )


    train_dataloader = DataLoader(
      datasets_train,
      batch_size=batch_size,
      # num_workers=num_workers,
      pin_memory=True,
    )

    valid_dataloader = DataLoader(
        datasets_valid,
        batch_size=batch_size,
        # num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        datasets_test,
        batch_size=batch_size,
        # num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader, test_dataloader