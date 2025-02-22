# steps/train.py
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
import logging
from sagemaker import Session


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


@step(
    instance_type="ml.m5.large",
    dependencies="requirements.txt"
)
def train(data_s3_path):
    """
    Trains an LSTM model on preprocessed stock data, evaluates it, saves the model to S3,
    and returns the S3 path and RMSE.
    """
    logger = setup_logging()
    logger.info(f"Starting training with data from {data_s3_path}")

    try:
        # Download preprocessed data from S3
        local_file = download_from_s3(data_s3_path)
        data = pd.read_csv(local_file).values  # Assuming columns: Open, High, Low, Close
        os.remove(local_file)  # Clean up
        logger.info(f"Loaded preprocessed data with shape {data.shape}")

        # Scale and sequence data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        X_all, y_all = create_sequences(data_scaled)
        train_size = int(0.8 * len(X_all))
        X_train, y_train = X_all[:train_size], y_all[:train_size]
        X_test, y_test = X_all[train_size:], y_all[train_size:]

        # Create dataloaders
        train_dataset = SPYDataset(X_train, y_train)
        test_dataset = SPYDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Model setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMTimeSeries(input_size=4, hidden_size=64, num_layers=1, output_size=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train the model
        train_model(model, train_loader, optimizer, criterion, epochs=10, device=device)

        # Evaluate the model
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
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        model_filename = f"model_{timestamp}_rmse_{rmse:.4f}.safetensors"
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict_cpu, model_filename)
        model_s3_path = upload_to_s3(open(model_filename, "rb").read(), get_default_bucket(), f"model/{model_filename}")
        os.remove(model_filename)  # Clean up
        logger.info(f"Model saved to {model_s3_path}")

        return model_s3_path, rmse
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e