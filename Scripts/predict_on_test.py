import os
import torch
from Scripts.config import Config
import pandas as pd
import torch.nn as nn
from Scripts.model import Model
from Scripts.dataset import DatasetRetriever
from Scripts.augmentations import testing_augmentations
from torch.utils.data import DataLoader


def predict_on_test_and_generate_submission_file(
    test_df: pd.DataFrame, model_path: str, use_tabular_features: bool = False
):
    """
    This function generates prediction probabilities on the
    test dataset and returns a submission.csv file.
    Args:
        test_df = test_dataframe.
        model_path = location where model state_dict is located.
        use_tabular_features: whether to use the tabular features
        or not.
    """
    test_dataset = DatasetRetriever(
        df=test_df,
        tabular_features=None,
        use_tabular_features=False,
        augmentations=testing_augmentations,
        is_test=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
    )
    test_predictions = []
    if torch.cuda.device_count() in (0, 1):
        model = Model().to(
            Config.DEVICE
        )
    elif torch.cuda.device_count() > 1:
        model = Model().to(
            Config.DEVICE
        )
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.inference_mode():
        for _, data in enumerate(test_dataloader):
            if use_tabular_features:
                data["image"], data["tabular_features"] = data["image"].to(
                    Config.DEVICE, dtype=torch.float
                ), data["tabular_features"].to(Config.DEVICE, dtype=torch.float)
                y_logits = model(data["image"], data["tabular_features"])
            else:
                data["image"] = data["image"].to(Config.DEVICE, dtype=torch.float)
                y_logits = model(data["image"]).squeeze(dim=0)
            test_probs = torch.sigmoid(y_logits).detach().cpu().numpy()
            test_predictions.extend(test_probs)
    submission_df = pd.read_csv(Config.submission_csv_path)
    test_predictions = [
        test_predictions[img].item() for img in range(len(test_predictions))
    ]
    submission_df["target"] = test_predictions
    submission_df.to_csv("../working/submission.csv", index=False)