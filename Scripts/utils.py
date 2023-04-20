import os
import torch
import random
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


def create_folds(train_df):
    """
    Function that folds in the training data and removes duplicate
    images from the training data.
    """
    train_df = train_df.loc[train_df["tfrecord"] != -1].reset_index(drop=True)
    train_df["fold"] = train_df["tfrecord"] % 5
    return train_df


def seed_everything(seed: int):
    """
    Function to set seed and to make reproducible results.
    Args:
        seed (int): like 42
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Directly borrowed from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(
        self,
        path: str,
        patience: int = 7,
        verbose: bool = False,
        delta: int = 0,
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(obj=model.state_dict(), f=self.path)
        self.val_loss_min = val_loss


def plot_loss_curves(results: dict):
    """
    Function to plot training & validation loss curves & validation AUC
    Args:
        results (dict): A dictionary of training loss, validation_loss &
        validation AUC score.
    """
    loss = results["train_loss"]
    valid_loss = results["valid_loss"]
    # Get the accuracy values of the results dictionary (training and test)
    valid_auc = results["valid_auc"]
    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))
    # Setup a plot
    plt.figure(figsize=(15, 7))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_auc, label="valid_auc")
    plt.title("AUC Score")
    plt.xlabel("Epochs")
    plt.legend()


class RareLabelCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Class to combine rare categories of a categorical variable
    where a category appearing less than a certain percentage
    (as a threshold).
    Example: a category/categories appearing less than 5% of the
    times are combined a single category called rare.
    """

    def __init__(self, variables: List, tol=0.05):
        """
        Args:
            variables (List): A list of variables for which we want
            to combine into rare categories.
            tol (int): A Threshold/Tolerance below which we want to
            consider a category of a feature as rare.
        """
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.tol = tol
        self.variables = variables

    def fit(self, x: pd.DataFrame):
        """
        This function learns all the values/categories & the
        percentage of times it appears in a feature in the
        dataset passed while using this method.

        Args:
            X : From this dataset the fit function learns and
            stores the number of times a category appears in
            the dataset
        """
        self.encoder_dict_ = {}
        for var in self.variables:
            t = pd.Series(x[var]).value_counts(normalize=True)
            self.encoder_dict_[var] = list(t[t >= self.tol].index)
        return self

    def transform(self, x: pd.DataFrame):
        """
        X (pd.DataFrame): Transform/Combines the categories of each
        features passed into the variables list on the dataset passed
        in this method and returns the transformed dataset.
        """
        x = x.copy()
        for var in self.variables:
            x[var] = np.where(x[var].isin(self.encoder_dict_[var]), x[var], "Other")
        return x


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """
    Class to handle outliers in a continous feature.
    """

    def __init__(
        self, variable: str, upper_quantile: float = None, lower_quantile: float = None
    ):
        """
        Args:
            variables (str): A variable to cap the upper and
            lower boundaries of.
            upper_quantile (float): A maximum value beyond which all the
            values of a feature are capped at.
            lower_quantile (float): A minimum value that are lower than
            of the feature are capped at.
        """
        if not isinstance(variable, str):
            raise ValueError("Variable should be a string type.")
        self.upper_quantile = upper_quantile
        self.variable = variable
        self.lower_quantile = lower_quantile

    def fit(self, x: pd.DataFrame):
        """
        This function learns the lower & upper quantiles of a feature
        present in the dataset x.
        """
        self.upper_quantile = x[self.variable].quantile(self.upper_quantile)
        self.lower_quantile = x[self.variable].quantile(self.lower_quantile)
        return self

    def transform(self, x: pd.DataFrame):
        """
        This function caps the upper and lower quantiles in the dataframe
        x with the values learnt in the dataframe passed in fit() method.
        """
        x = x.copy()
        x[self.variable] = np.where(
            x[self.variable] > self.upper_quantile,
            self.upper_quantile,
            np.where(
                x[self.variable] < self.lower_quantile,
                self.lower_quantile,
                x[self.variable],
            ),
        )
        return x
