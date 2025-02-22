# cloud/steps/deploy.py
from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, upload_to_s3, setup_logging, download_from_s3, LSTMTimeSeries
import boto3
import torch
from safetensors.torch import load_file
import logging
from sagemaker import get_execution_role, Session
from sagemaker.pytorch import PyTorchModel
import os
import tarfile
from datetime import datetime

@step(instance_type="ml.m5.large")  # No dependencies="steps/" as per your config
def deploy(eval_output):
    """
    Deploys the model to a SageMaker endpoint if deploy_flag is True, updates the current model in S3,
    and returns the endpoint name or None.
    Args:
        eval_output (tuple): Tuple (deploy_flag: bool, model_s3_path: str, hidden_size: int, num_layers: int) from eval step.
    Returns:
        str or None: SageMaker endpoint name if deployed, else None.
    """
    logger = setup_logging()
    deploy_flag, model_s3_path, hidden_size, num_layers = eval_output
    logger.info(f"Received eval output: deploy_flag={deploy_flag}, model_s3_path={model_s3_path}, hidden_size={hidden_size}, num_layers={num_layers}")

    try:
        if not deploy_flag:
            logger.info("Deploy flag is False; skipping deployment.")
            return None

        # Load the model from S3 with tuned hyperparameters
        local_model_path = download_from_s3(model_s3_path)
        model = LSTMTimeSeries(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
        state_dict = load_file(local_model_path)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_s3_path}")

        # Use a unique subdirectory to avoid recursion issues
        model_dir = f"/tmp/deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "model.safetensors")
        torch.save(model.state_dict(), model_file)

        # Write inference script with tuned hyperparameters
        inference_script = f"""
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
    model = LSTMTimeSeries(input_size=4, hidden_size={hidden_size}, num_layers={num_layers}, output_size=1)
    model_path = os.path.join(model_dir, "model.safetensors")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Expecting [seq_len, 4]
        return input_data
    raise ValueError(f"Unsupported content type: {{request_content_type}}")

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        output = model(input_data)
    return output.cpu().numpy()

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps(prediction.tolist())
    raise ValueError(f"Unsupported content type: {{content_type}}")
"""
        inference_file = os.path.join(model_dir, "inference.py")
        with open(inference_file, "w") as f:
            f.write(inference_script)

        # Package model artifacts into a tar.gz file
        tar_path = os.path.join(model_dir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="model.safetensors")
            tar.add(inference_file, arcname="inference.py")

        # Upload model artifacts to S3
        bucket = get_default_bucket()
        model_artifact_s3 = upload_to_s3(open(tar_path, "rb").read(), bucket, "model/model.tar.gz")
        logger.info(f"Model artifacts uploaded to {model_artifact_s3}")

        # Deploy to SageMaker endpoint
        sagemaker_session = Session()
        pytorch_model = PyTorchModel(
            model_data=model_artifact_s3,
            role=get_execution_role(),
            framework_version="2.0",
            py_version="py310",
            entry_point="inference.py",
            source_dir=model_dir,  # Use unique subdirectory
            sagemaker_session=sagemaker_session
        )
        endpoint_name = f"stock-pipeline-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large",
            endpoint_name=endpoint_name
        )
        logger.info(f"Model deployed to endpoint: {endpoint_name}")

        # Update current model in S3
        current_model_path = f"s3://{bucket}/model/current_model.safetensors"
        s3_client = boto3.client('s3')
        s3_client.copy_object(
            Bucket=bucket,
            Key="model/current_model.safetensors",
            CopySource={'Bucket': bucket, 'Key': model_s3_path.split('/', 3)[-1]}
        )
        logger.info(f"Updated current model at {current_model_path}")

        # Clean up temporary files
        os.remove(local_model_path)
        shutil.rmtree(model_dir)  # Remove entire directory
        return endpoint_name

    except Exception as e:
        logger.error(f"Error during deployment: {e}")
        raise e

    finally:
        # Ensure cleanup even on failure
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        if os.path.exists(local_model_path):
            os.remove(local_model_path)