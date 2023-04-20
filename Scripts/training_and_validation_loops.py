import torch
import numpy as np
from tqdm import tqdm
import torch.cuda.amp as amp
from Scripts.utils import EarlyStopping
from sklearn.metrics import roc_auc_score


def train_one_epoch(
    model, dataloader, loss_fn, optimizer, device, scaler, use_tabular_features=False
):
    """
    Function takes a model instance, dataloader, loss function, an optimizer, device
    (on which device should you want to run the model on i.e., GPU or CPU)
    , scaler (for mixed precision) and whether to use tabular features or not.
    This function runs/passes the images and tabular features for a single epoch
    and returns the loss value on training dataset.
    """
    train_loss = 0
    model.train()
    for data in dataloader:
        optimizer.zero_grad()
        if use_tabular_features:
            data["image"], data["tabular_features"], data["targets"] = (
                data["image"].to(device, dtype=torch.float),
                data["tabular_features"].to(device, dtype=torch.float),
                data["targets"].to(device, dtype=torch.float),
            )
            with amp.autocast():
                y_logits = model(data["image"], data["tabular_features"]).squeeze(dim=0)
                loss = loss_fn(y_logits, data["targets"].view(-1, 1))
        else:
            data["image"], data["targets"] = data["image"].to(
                device, dtype=torch.float
            ), data["targets"].to(device, dtype=torch.float)
            with amp.autocast():
                y_logits = model(data["image"]).squeeze(dim=0)
                loss = loss_fn(y_logits, data["targets"].view(-1, 1))
        train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    train_loss = train_loss / len(dataloader)
    return train_loss


def validate_one_epoch(model, dataloader, loss_fn, device, use_tabular_features=False):
    """
    Function takes a model instance, dataloader, loss function, device
    (on which device should you want to run the model on i.e., GPU or CPU)
    and whether to use tabular features or not.
    This function runs/passes the images and tabular features for a single epoch
    and returns the loss value & final predictions on the validation dataset.
    """
    valid_loss, final_predictions = 0, []
    model.eval()
    with torch.inference_mode():
        for data in dataloader:
            if use_tabular_features:
                data["image"], data["tabular_features"], data["targets"] = (
                    data["image"].to(device, dtype=torch.float),
                    data["tabular_features"].to(device, dtype=torch.float),
                    data["targets"].to(device, dtype=torch.float),
                )
                y_logits = model(data["image"], data["tabular_features"]).squeeze(dim=0)
            else:
                data["image"], data["targets"] = data["image"].to(
                    device, dtype=torch.float
                ), data["targets"].to(device, dtype=torch.float)
                y_logits = model(data["image"]).squeeze(dim=0)
            loss = loss_fn(y_logits, data["targets"].view(-1, 1))
            valid_loss += loss.item()
            valid_probs = torch.sigmoid(y_logits).detach().cpu().numpy()
            final_predictions.extend(valid_probs)
    valid_loss = valid_loss / len(dataloader)
    return valid_loss, final_predictions


def train(
    model,
    train_dataloader,
    valid_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    device,
    scaler,
    epochs,
    es_patience,
    model_save_path,
    validation_targets,
):
    """
    This function takes a model instance, training dataloader,
    validation dataloader, loss_fn, optimizer, scheduler, device,
    scaler (object, for mixed precision), epochs (for how many epochs
    to run the model), es_patience (number of epochs to wait after which
    the model should stop training), model_save_path (where to save the
    model to), validation_targets (used for the calculation of the AUC
    score) and returns a dictionary object which has training loss,
    validation loss and validation AUC score.
    """
    results = {"train_loss": [], "valid_loss": [], "valid_auc": []}

    early_stopping = EarlyStopping(
        patience=es_patience, verbose=True, path=model_save_path
    )

    for epoch in tqdm(range(epochs)):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_tabular_features=False,
        )

        valid_loss, valid_predictions = validate_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device,
            use_tabular_features=False,
        )

        valid_predictions = np.vstack(valid_predictions).ravel()

        valid_auc = roc_auc_score(y_score=valid_predictions, y_true=validation_targets)
        scheduler.step(valid_auc)

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early Stopping")
            break

        model.load_state_dict(torch.load(model_save_path))
        print(
            f"Epoch : {epoch+1} | "
            f"train_loss : {train_loss:.4f} | "
            f"valid_loss : {valid_loss:.4f} | "
            f"valid_auc : {valid_auc:.4f} "
        )
        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)
        results["valid_auc"].append(valid_auc)
    return results
