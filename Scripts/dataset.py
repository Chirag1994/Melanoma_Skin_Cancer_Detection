import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from PIL import ImageFile
from typing import List, Callable

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetRetriever(nn.Module):
    """
    Dataset class to read the images and tabular features from a
    dataframe and returns the dictionary.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tabular_features: List[str] = None,
        use_tabular_features: bool = False,
        augmentations: Callable = None,
        is_test: bool = False,
    ):
        """ """
        self.df = df
        self.tabular_features = tabular_features
        self.use_tabular_features = use_tabular_features
        self.augmentations = augmentations
        self.is_test = is_test

    def __len__(self):
        """
        Function returns the number of images in a dataframe.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Function the takes an images and it's corresponding
        tabular/meta features & target feature (for training
        and validation) and returns a dictionary, otherwise,
        for test dataset it only returns a dictionary of
        an image and tabular features.
        """
        image_path = self.df["image_path"].iloc[index]
        image = Image.open(image_path)
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float)
        if self.use_tabular_features:
            if len(self.tabular_features) > 0 and self.is_test is False:
                tabular_features = np.array(
                    self.df.iloc[index][self.tabular_features].values, dtype=np.float32
                )
                targets = self.df.target[index]
                return {
                    "image": image,
                    "tabular_features": tabular_features,
                    "targets": torch.tensor(targets, dtype=torch.long),
                }
            elif len(self.tabular_features) > 0 and self.is_test is True:
                tabular_features = np.array(
                    self.df.iloc[index][self.tabular_features].values, dtype=np.float32
                )
                return {"image": image, "tabular_features": tabular_features}
        else:
            if self.is_test is False:
                targets = self.df.target[index]
                return {
                    "image": image,
                    "targets": torch.tensor(targets, dtype=torch.long),
                }
            elif self.is_test is True:
                return {"image": image}
