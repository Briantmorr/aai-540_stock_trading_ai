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

class SPYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, window_size=30, target_step=1):
    """Create sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - window_size - target_step + 1):
        X_seq = data[i:i + window_size]
        y_seq = data[i + window_size + target_step - 1]
        X.append(X_seq)
        y.append(y_seq[3])  # Predict 'Close'
    return np.array(X), np.array(y)


def train_model(model, train_loader, optimizer, criterion, epochs, device):
    """Train the LSTM model."""
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
    """Evaluate the model on the test set."""
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
    """Prepare data for hyperparameter tuning with train/val/test split."""
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
    """Optuna objective function to minimize RMSE on validation set."""
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # Load and split data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data_for_tuning(
        data,
        sequence_length=sequence_length,
        prediction_days=prediction_days,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # Create dataloaders for tuning
    train_dataset = SPYDataset(X_train, y_train)
    val_dataset = SPYDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 4  # Open, High, Low, Close
    output_size = 1
    model = LSTMTimeSeries(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
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

    # Inverse scale for RMSE
    pred_close_scaled = np.zeros((len(val_predictions), 4))
    act_close_scaled = np.zeros((len(val_actuals), 4))
    pred_close_scaled[:, 3] = val_predictions.flatten()
    act_close_scaled[:, 3] = val_actuals.flatten()
    pred_close = scaler.inverse_transform(pred_close_scaled)[:, 3]
    act_close = scaler.inverse_transform(act_close_scaled)[:, 3]

    rmse_val = float(np.sqrt(np.mean((pred_close - act_close) ** 2)))
    return rmse_val

def tune_hyperparams(data, sequence_length=30, prediction_days=1, n_epochs=10, val_ratio=0.1, test_ratio=0.1, n_trials=20):
    """Run Optuna hyperparameter tuning."""
    logger = setup_logging()
    logger.info("Starting hyperparameter tuning with Optuna...")

    def optuna_objective(trial):
        return objective(
            trial,
            data,
            sequence_length=sequence_length,
            prediction_days=prediction_days,
            n_epochs=n_epochs,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_objective, n_trials=n_trials)
    logger.info(f"Best trial: {study.best_trial}")
    logger.info(f"Best RMSE: {study.best_value}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params


@step(instance_type="ml.m5.large")
def train(data_s3_path):
    """
    Trains an LSTM model with hyperparameter tuning on preprocessed stock data,
    evaluates it, saves the model to S3, and returns the S3 path,
    RMSE, and tuned hyperparameters.
    """
    logger = setup_logging()
    logger.info(f"Starting training with data from {data_s3_path}")

    try:
        local_file = download_from_s3(data_s3_path)
        data = pd.read_csv(local_file).values  # Expecting Open, High, Low, Close (4 columns)
        os.remove(local_file)
        logger.info(f"Loaded preprocessed data with shape {data.shape}")

        best_params = tune_hyperparams(
            data,
            sequence_length=30,
            prediction_days=1,
            n_epochs=10,
            val_ratio=0.1,
            test_ratio=0.1,
            n_trials=20
        )
        hidden_size = best_params["hidden_size"]
        num_layers = best_params["num_layers"]
        lr = best_params["lr"]
        batch_size = best_params["batch_size"]
        logger.info(f"Using best hyperparameters: {best_params}")

        # Scale and sequence data with final parameters
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        X_all, y_all = create_sequences(data_scaled, window_size=30, target_step=1)
        train_size = int(0.8 * len(X_all))
        X_train, y_train = X_all[:train_size], y_all[:train_size]
        X_test, y_test = X_all[train_size:], y_all[train_size:]

        # Create dataloaders with tuned batch_size
        train_dataset = SPYDataset(X_train, y_train)
        test_dataset = SPYDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Model setup with tuned parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMTimeSeries(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_model(model, train_loader, optimizer, criterion, epochs=10, device=device)

        predictions, actuals = evaluate_model(model, test_loader, device)

        # Inverse scale predictions for RMSE
        pred_scaled = np.zeros((len(predictions), 4))
        act_scaled = np.zeros((len(actuals), 4))
        pred_scaled[:, 3], act_scaled[:, 3] = predictions.flatten(), actuals.flatten()
        pred_close = scaler.inverse_transform(pred_scaled)[:, 3]
        act_close = scaler.inverse_transform(act_scaled)[:, 3]
        rmse = float(np.sqrt(np.mean((pred_close - act_close) ** 2)))
        logger.info(f"Training completed with RMSE: {rmse}")

        # Save model to S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"model_{timestamp}_rmse_{rmse:.4f}.safetensors"
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict_cpu, model_filename)
        model_s3_path = upload_to_s3(open(model_filename, "rb").read(), get_default_bucket(), f"model/{model_filename}")
        os.remove(model_filename)
        logger.info(f"Model saved to {model_s3_path}")

        # Return model_s3_path, rmse, and tuned hyperparameters
        return model_s3_path, rmse, hidden_size, num_layers, batch_size, lr

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e