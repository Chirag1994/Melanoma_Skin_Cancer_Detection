import os
import torch
from Scripts.config import Config
import pandas as pd
import torch.nn as nn
from Scripts.model import Model
import torch.cuda.amp as amp
from Scripts.utils import create_folds
from Scripts.utils import seed_everything
from Scripts.dataset import DatasetRetriever
from timeit import default_timer as timer
from Scripts.training_and_validation_loops import train
from torch.utils.data import Dataset, DataLoader
from Scripts.augmentations import training_augmentations, validation_augmentations


def run_model(fold, train_df):
    train_df = create_folds(train_df=train_df)
    train_data = train_df.loc[train_df["fold"] != fold].reset_index(drop=True)
    valid_data = train_df.loc[train_df["fold"] == fold].reset_index(drop=True)
    validation_targets = valid_data["target"]
    train_dataset = DatasetRetriever(
        df=train_data,
        tabular_features=None,
        use_tabular_features=False,
        augmentations=training_augmentations,
        is_test=False,
    )
    valid_dataset = DatasetRetriever(
        df=valid_data,
        tabular_features=None,
        use_tabular_features=False,
        augmentations=validation_augmentations,
        is_test=False,
    )
    training_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
    )
    validation_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
    )
    seed_everything(Config.RANDOM_STATE)
    if torch.cuda.device_count() in (0, 1):
        model = Model().to(Config.DEVICE)
    elif torch.cuda.device_count() > 1:
        model = Model().to(Config.DEVICE)
        model = nn.DataParallel(model)
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.2,
        patience=2,
        threshold=1e-3,
        verbose=True,
    )
    scaler = amp.GradScaler()
    start_time = timer()
    model_save_path = f"../working/Models/efficientnet_b5_checkpoint_fold_{fold}.pt"
    model_results = train(
        model=model,
        train_dataloader=training_dataloader,
        valid_dataloader=validation_dataloader,
        loss_fn=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        scaler=scaler,
        epochs=Config.EPOCHS,
        es_patience=2,
        model_save_path=model_save_path,
        validation_targets=validation_targets,
    )
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
