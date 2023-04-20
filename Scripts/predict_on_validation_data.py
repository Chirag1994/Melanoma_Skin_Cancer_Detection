import os
import torch
from Scripts.config import Config
import pandas as pd
import torch.nn as nn
from Scripts.model import Model
from Scripts.dataset import DatasetRetriever
from Scripts.augmentations import validation_augmentations
from torch.utils.data import DataLoader


def predict_on_validation_dataset(
    validation_df: pd.DataFrame, model_path: str, use_tabular_features: bool = False
):
    """
    This function generates prediction probabilities on the
    validation dataset and returns a submission.csv file.
    Args:
        validation_dataset = validation_dataframe.
        model_path = location where model state_dict is located.
        use_tabular_features: whether to use the tabular features
        or not.
    """
    valid_dataset = DatasetRetriever(
        df=validation_df,
        tabular_features=None,
        use_tabular_features=False,
        augmentations=validation_augmentations,
        is_test=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
    )
    valid_predictions = []
    if torch.cuda.device_count() in (0, 1):
        model = Model().to(Config.DEVICE)
    elif torch.cuda.device_count() > 1:
        model = Model().to(Config.DEVICE)
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.inference_mode():
        for _, data in enumerate(valid_dataloader):
            if use_tabular_features:
                data["image"], data["tabular_features"] = data["image"].to(
                    Config.DEVICE, dtype=torch.float
                ), data["tabular_features"].to(Config.DEVICE, dtype=torch.float)
                y_logits = model(data["image"], data["tabular_features"])
            else:
                data["image"] = data["image"].to(Config.DEVICE, dtype=torch.float)
                y_logits = model(data["image"]).squeeze(dim=0)
            valid_probs = torch.sigmoid(y_logits).detach().cpu().numpy()
            valid_predictions.extend(valid_probs)
    valid_predictions = [
        valid_predictions[img].item() for img in range(len(valid_predictions))
    ]
    return valid_predictions
