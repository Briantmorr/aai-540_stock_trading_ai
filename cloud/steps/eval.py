from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, setup_logging, LSTMTimeSeries
import boto3
import torch
from safetensors.torch import load
import io
import logging
from sagemaker import Session


def load_model_from_s3_direct(s3_path):
    """Load a model directly from S3 without writing to local disk."""
    logger = setup_logging()
    s3_client = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    model_bytes = response['Body'].read()
    state_dict = load(model_bytes)  # Deserialize directly from bytes
    logger.info(f"Loaded model from {s3_path}")
    return state_dict


def get_rmse_from_filename(s3_path):
    """Extract RMSE from the model filename."""
    try:
        rmse_str = s3_path.split("rmse_")[-1].split(".safetensors")[0]
        return float(rmse_str)
    except (IndexError, ValueError):
        return None


# @step(
#     instance_type="ml.m5.large",
#     dependencies="requirements.txt"
# )
def eval(train_output):
    """
    Evaluates the trained model against the current deployed model by comparing RMSE.
    Returns a flag indicating whether to deploy the new model and the model S3 path.
    Args:
        train_output (tuple): Tuple (model_s3_path, rmse) from the train step.
    Returns:
        tuple: (deploy_flag, model_s3_path) where deploy_flag is a boolean and
               model_s3_path is the S3 path to the new model.
    """
    logger = setup_logging()
    model_s3_path, new_rmse = train_output
    logger.info(f"Evaluating model from {model_s3_path} with RMSE: {new_rmse}")

    try:
        # Load the new model (optional verification)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_model = LSTMTimeSeries(input_size=4, hidden_size=64, num_layers=1, output_size=1).to(device)
        new_state_dict = load_model_from_s3_direct(model_s3_path)
        new_model.load_state_dict(new_state_dict)

        # Get the current model’s S3 path
        bucket = get_default_bucket()
        current_model_path = f"s3://{bucket}/model/current_model.safetensors"

        # Try to load the current model and extract its RMSE
        try:
            current_state_dict = load_model_from_s3_direct(current_model_path)
            current_model = LSTMTimeSeries(input_size=4, hidden_size=64, num_layers=1, output_size=1).to(device)
            current_model.load_state_dict(current_state_dict)
            current_rmse = get_rmse_from_filename(current_model_path)
            if current_rmse is None:
                logger.warning(f"Could not extract RMSE from {current_model_path}, assuming infinity")
                current_rmse = float("inf")
        except Exception as e:
            logger.info(f"No current model found or error loading {current_model_path}: {e}. Assuming new model is better.")
            current_rmse = float("inf")

        # Decide whether to deploy the new model
        deploy_flag = new_rmse < current_rmse
        logger.info(f"New RMSE: {new_rmse}, Current RMSE: {current_rmse}, Deploy: {deploy_flag}")

        return deploy_flag, model_s3_path
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e