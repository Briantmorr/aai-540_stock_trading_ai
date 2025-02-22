# cloud/steps/eval.py
from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, setup_logging, download_from_s3, LSTMTimeSeries
import boto3
import torch
from safetensors.torch import load_file
import os
import logging

def get_rmse_from_filename(s3_path):
    """Extract RMSE from the model filename."""
    try:
        rmse_str = s3_path.split("rmse_")[-1].split(".safetensors")[0]
        return float(rmse_str)
    except (IndexError, ValueError):
        return None

@step(instance_type="ml.m5.large")  # No dependencies="steps/" as per your config
def eval(train_output):
    """
    Evaluates the trained model against the current deployed model by comparing RMSE.
    Returns a flag indicating whether to deploy the new model, the model S3 path, and tuned hyperparameters.
    
    Args:
        train_output (tuple): Tuple (model_s3_path, rmse, hidden_size, num_layers, batch_size, lr) from train step.
    
    Returns:
        tuple: (deploy_flag, model_s3_path, hidden_size, num_layers) for deploy step.
    """
    logger = setup_logging()
    model_s3_path, new_rmse, hidden_size, num_layers, _, _ = train_output  # Unpack tuned hyperparameters
    logger.info(f"Evaluating model from {model_s3_path} with RMSE: {new_rmse}")

    try:
        # Load the new model with tuned hyperparameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_model = LSTMTimeSeries(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
        local_model_path = download_from_s3(model_s3_path)
        new_state_dict = load_file(local_model_path)
        new_model.load_state_dict(new_state_dict)
        logger.info(f"Loaded model from {model_s3_path}")

        # Get the current model’s S3 path
        bucket = get_default_bucket()
        current_model_path = f"s3://{bucket}/model/current_model.safetensors"

        # Try to load the current model and extract its RMSE
        try:
            current_local_path = download_from_s3(current_model_path)
            current_state_dict = load_file(current_local_path)
            current_model = LSTMTimeSeries(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
            current_model.load_state_dict(current_state_dict)
            current_rmse = get_rmse_from_filename(current_model_path)
            if current_rmse is None:
                logger.warning(f"Could not extract RMSE from {current_model_path}, assuming infinity")
                current_rmse = float("inf")
            os.remove(current_local_path)
        except Exception as e:
            logger.info(f"No current model found or error loading {current_model_path}: {e}. Assuming new model is better.")
            current_rmse = float("inf")

        # Decide whether to deploy the new model
        deploy_flag = new_rmse < current_rmse
        logger.info(f"New RMSE: {new_rmse}, Current RMSE: {current_rmse}, Deploy: {deploy_flag}")

        os.remove(local_model_path)
        return deploy_flag, model_s3_path, hidden_size, num_layers

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e