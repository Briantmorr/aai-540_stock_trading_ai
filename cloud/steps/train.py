from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, upload_to_s3, setup_logging, download_from_s3, LSTMTimeSeries
import boto3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from safetensors.torch import save_file
from datetime import datetime
import os
import optuna
import logging
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.metadata_properties import MetadataProperties  # Added for metadata properties
import json
import tarfile
import shutil


class SPYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, window_size=30, target_step=1):
    X, y = [], []
    for i in range(len(data) - window_size - target_step + 1):
        X_seq = data[i:i + window_size]
        y_seq = data[i + window_size + target_step - 1]
        X.append(X_seq)
        y.append(y_seq[3])  # Predict 'Close'
    return np.array(X), np.array(y)


def train_model(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    logger = setup_logging()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.numpy().reshape(-1, 1))
    return np.vstack(predictions), np.vstack(actuals)


def load_data_for_tuning(data, sequence_length=30, prediction_days=1, val_ratio=0.1, test_ratio=0.1):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X_all, y_all = create_sequences(data_scaled, window_size=sequence_length, target_step=prediction_days)
    dataset_size = len(X_all)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    train_size = dataset_size - val_size - test_size
    X_train, X_val, X_test = X_all[:train_size], X_all[train_size:train_size+val_size], X_all[train_size+val_size:]
    y_train, y_val, y_test = y_all[:train_size], y_all[train_size:train_size+val_size], y_all[train_size+val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def objective(trial, data, sequence_length=30, prediction_days=1, n_epochs=10, val_ratio=0.1, test_ratio=0.1):
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data_for_tuning(
        data, sequence_length, prediction_days, val_ratio, test_ratio
    )

    train_dataset = SPYDataset(X_train, y_train)
    val_dataset = SPYDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTimeSeries(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    val_predictions, val_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            val_predictions.append(outputs.cpu().numpy())
            val_actuals.append(y_batch.numpy().reshape(-1, 1))

    val_predictions = np.vstack(val_predictions)
    val_actuals = np.vstack(val_actuals)
    pred_close_scaled = np.zeros((len(val_predictions), 4))
    act_close_scaled = np.zeros((len(val_actuals), 4))
    pred_close_scaled[:, 3] = val_predictions.flatten()
    act_close_scaled[:, 3] = val_actuals.flatten()
    pred_close = scaler.inverse_transform(pred_close_scaled)[:, 3]
    act_close = scaler.inverse_transform(act_close_scaled)[:, 3]
    rmse_val = float(np.sqrt(np.mean((pred_close - act_close) ** 2)))
    return rmse_val


def tune_hyperparams(data, sequence_length=30, prediction_days=1, n_epochs=10, val_ratio=0.1, test_ratio=0.1, n_trials=20):
    logger = setup_logging()
    logger.info("Starting hyperparameter tuning with Optuna...")

    def optuna_objective(trial):
        return objective(trial, data, sequence_length, prediction_days, n_epochs, val_ratio, test_ratio)

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=n_trials)
    logger.info(f"Best trial: {study.best_trial}")
    logger.info(f"Best RMSE: {study.best_value}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params


@step(instance_type="ml.m5.large")
def train(data_s3_path):
    logger = setup_logging()
    logger.info(f"Starting training with data from {data_s3_path}")

    try:
        local_file = download_from_s3(data_s3_path)
        data = pd.read_csv(local_file).values
        os.remove(local_file)
        logger.info(f"Loaded preprocessed data with shape {data.shape}")

        best_params = tune_hyperparams(data)
        hidden_size = best_params["hidden_size"]
        num_layers = best_params["num_layers"]
        lr = best_params["lr"]
        batch_size = best_params["batch_size"]
        logger.info(f"Using best hyperparameters: {best_params}")

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        X_all, y_all = create_sequences(data_scaled)
        train_size = int(0.8 * len(X_all))
        X_train, y_train = X_all[:train_size], y_all[:train_size]
        X_test, y_test = X_all[train_size:], y_all[train_size:]

        train_dataset = SPYDataset(X_train, y_train)
        test_dataset = SPYDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMTimeSeries(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_model(model, train_loader, optimizer, criterion, epochs=10, device=device)

        predictions, actuals = evaluate_model(model, test_loader, device)
        pred_scaled = np.zeros((len(predictions), 4))
        act_scaled = np.zeros((len(actuals), 4))
        pred_scaled[:, 3], act_scaled[:, 3] = predictions.flatten(), actuals.flatten()
        pred_close = scaler.inverse_transform(pred_scaled)[:, 3]
        act_close = scaler.inverse_transform(act_scaled)[:, 3]
        rmse = float(np.sqrt(np.mean((pred_close - act_close) ** 2)))
        logger.info(f"Training completed with RMSE: {rmse}")

        model_dir = f"/tmp/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "model.safetensors")
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict_cpu, model_file)

        inference_script = """
import torch
import torch.nn as nn
import os
import json

class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(model_dir, "hyperparameters.json"), "r") as f:
        hyperparams = json.load(f)
    model = LSTMTimeSeries(
        input_size=4,
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_layers"],
        output_size=1
    )
    model_path = os.path.join(model_dir, "model.safetensors")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        return input_data
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
    return output.cpu().numpy()

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {content_type}")
"""
        inference_file = os.path.join(model_dir, "inference.py")
        with open(inference_file, "w") as f:
            f.write(inference_script)

        hyperparams = {"hidden_size": hidden_size, "num_layers": num_layers}
        with open(os.path.join(model_dir, "hyperparameters.json"), "w") as f:
            json.dump(hyperparams, f)

        tar_path = os.path.join(model_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="model.safetensors")
            tar.add(inference_file, arcname="inference.py")
            tar.add(os.path.join(model_dir, "hyperparameters.json"), arcname="hyperparameters.json")

        bucket = get_default_bucket()
        s3_model_path = f"model-registry/stock-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}/model.tar.gz"
        model_s3_uri = upload_to_s3(open(tar_path, "rb").read(), bucket, s3_model_path)
        logger.info(f"Model artifacts uploaded to {model_s3_uri}")

        sagemaker_session = sagemaker.Session()
        model_package_group_name = "StockPredictionModelGroup"
        model = PyTorchModel(
            model_data=model_s3_uri,
            role=sagemaker.get_execution_role(),
            framework_version="2.0",
            py_version="py310",
            entry_point="inference.py",
            source_dir=model_dir,
            sagemaker_session=sagemaker_session
        )
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f"s3://{bucket}/metrics/rmse-{rmse}.json",
                content_type="application/json"
            )
        )
        custom_metadata = {
            "hidden_size": str(hidden_size),
            "num_layers": str(num_layers)
        }
        model_package = model.register(
            model_package_group_name=model_package_group_name,
            approval_status="PendingManualApproval",
            model_metrics=model_metrics,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            description=f"Stock prediction model with RMSE {rmse}",
            customer_metadata_properties=custom_metadata  # Use customer_metadata_properties for custom metadata
        )
        model_package_arn = model_package.model_package_arn
        logger.info(f"Model registered with ARN: {model_package_arn}")

        shutil.rmtree(model_dir)
        return model_package_arn, rmse

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e
    finally:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)