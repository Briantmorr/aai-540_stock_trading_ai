from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, setup_logging, download_from_s3, LSTMTimeSeries
import boto3
import torch
from safetensors.torch import load_file
import os
import logging
import sagemaker
import tarfile  


@step(instance_type="ml.m5.large")
def eval(train_output):
    """
    Evaluates the trained model against the latest approved model in the SageMaker Model Registry.
    Returns a flag indicating whether to deploy the new model and the model package ARN.
    
    Args:
        train_output (tuple): Tuple (model_package_arn, rmse) from train step.
    
    Returns:
        tuple: (deploy_flag, model_package_arn).
    """
    logger = setup_logging()
    model_package_arn, new_rmse = train_output
    logger.info(f"Evaluating model with ARN {model_package_arn} and RMSE: {new_rmse}")

    # Initialize file paths to None for safe cleanup
    new_local_tar = None
    new_model_path = None
    current_local_tar = None
    current_model_path = None

    try:
        sagemaker_session = sagemaker.Session()
        sm_client = boto3.client("sagemaker")

        new_model_desc = sm_client.describe_model_package(ModelPackageName=model_package_arn)
        new_model_s3_uri = new_model_desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        new_metadata = new_model_desc.get("CustomerMetadataProperties", {})
        new_hidden_size = int(new_metadata.get("hidden_size", "64"))
        new_num_layers = int(new_metadata.get("num_layers", "1"))

        new_local_tar = download_from_s3(new_model_s3_uri)
        with tarfile.open(new_local_tar, "r:gz") as tar:
            tar.extract("model.safetensors", path=os.path.dirname(new_local_tar))
        new_model_path = os.path.join(os.path.dirname(new_local_tar), "model.safetensors")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        new_model = LSTMTimeSeries(input_size=4, hidden_size=new_hidden_size, num_layers=new_num_layers, output_size=1).to(device)
        new_state_dict = load_file(new_model_path)
        new_model.load_state_dict(new_state_dict)
        logger.info(f"Loaded new model from {new_model_path}")

        model_package_group_name = "StockPredictionModelGroup"
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )
        current_rmse = float("inf")
        if response.get("ModelPackageSummaryList"):
            current_model_package = response["ModelPackageSummaryList"][0]
            current_model_arn = current_model_package["ModelPackageArn"]
            current_model_desc = sm_client.describe_model_package(ModelPackageName=current_model_arn)
            current_model_s3_uri = current_model_desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
            current_metadata = current_model_desc.get("CustomerMetadataProperties", {})
            current_hidden_size = int(current_metadata.get("hidden_size", "64"))
            current_num_layers = int(current_metadata.get("num_layers", "1"))

            current_local_tar = download_from_s3(current_model_s3_uri)
            with tarfile.open(current_local_tar, "r:gz") as tar:
                tar.extract("model.safetensors", path=os.path.dirname(current_local_tar))
            current_model_path = os.path.join(os.path.dirname(current_local_tar), "model.safetensors")

            current_model = LSTMTimeSeries(input_size=4, hidden_size=current_hidden_size, num_layers=current_num_layers, output_size=1).to(device)
            current_state_dict = load_file(current_model_path)
            current_model.load_state_dict(current_state_dict)
            logger.info(f"Loaded current approved model from {current_model_path}")

            current_rmse_str = current_model_desc.get("ModelPackageDescription", "").split("RMSE ")[-1] or "inf"
            current_rmse = float(current_rmse_str) if current_rmse_str != "inf" else float("inf")
        else:
            logger.info("No approved model found in registry.")

        # Decide whether to approve/deploy the new model
        deploy_flag = new_rmse < current_rmse
        logger.info(f"New RMSE: {new_rmse}, Current RMSE: {current_rmse}, Deploy: {deploy_flag}")

        return deploy_flag, model_package_arn

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise e
    finally:
        for path in [new_local_tar, new_model_path, current_local_tar, current_model_path]:
            if path and os.path.exists(path):
                os.remove(path)