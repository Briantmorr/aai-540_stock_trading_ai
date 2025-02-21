import boto3
from sagemaker import Session
import logging
import torch
import torch.nn as nn
import os


def get_sagemaker_session():
    return Session()


def get_default_bucket():
    return get_sagemaker_session().default_bucket()


def upload_to_s3(data, bucket, key):
    s3_client = boto3.client('s3')
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
    return f"s3://{bucket}/{key}"


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def download_from_s3(s3_path):
    """Download a file from S3 and return the local path."""
    s3_client = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    local_file = f"/tmp/{os.path.basename(key)}"
    s3_client.download_file(bucket, key, local_file)
    return local_file


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