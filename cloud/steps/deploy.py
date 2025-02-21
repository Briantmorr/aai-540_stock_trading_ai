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

# @step(
#     instance_type="ml.m5.large",
#     dependencies="dependencies.txt"  
# )
def deploy(eval_output):
    """
    Deploys the model to a SageMaker endpoint if deploy_flag is True, updates the current model in S3,
    and returns the endpoint name or None.
    Args:
        eval_output (tuple): Tuple (deploy_flag: bool, model_s3_path: str) from eval step.
    Returns:
        str or None: SageMaker endpoint name if deployed, else None.
    """
    logger = setup_logging()
    deploy_flag, model_s3_path = eval_output
    logger.info(f"Received eval output: deploy_flag={deploy_flag}, model_s3_path={model_s3_path}")

    try:
        if not deploy_flag:
            logger.info("Deploy flag is False; skipping deployment.")
            return None

        # Load the model from S3
        local_model_path = download_from_s3(model_s3_path)
        model = LSTMTimeSeries(input_size=4, hidden_size=64, num_layers=1, output_size=1)
        state_dict = load_file(local_model_path)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_s3_path}")

        # Prepare model artifacts for SageMaker deployment
        model_dir = "/tmp/model"
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, "model.safetensors")
        torch.save(model.state_dict(), model_file)  # SageMaker expects a .pth or similar file

        # Use the existing inference.py from steps/
        inference_script_path = os.path.join(os.path.dirname(__file__), "inference.py")
        if not os.path.exists(inference_script_path):
            raise FileNotFoundError(f"inference.py not found at {inference_script_path}")

        # Package model artifacts into a tar.gz file
        tar_path = "/tmp/model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="model.safetensors")
            tar.add(inference_script_path, arcname="inference.py")

        # Upload model artifacts to S3
        bucket = get_default_bucket()
        model_artifact_s3 = upload_to_s3(open(tar_path, "rb").read(), bucket, "model/model.tar.gz")
        logger.info(f"Model artifacts uploaded to {model_artifact_s3}")

        # Clean up temporary files
        os.remove(local_model_path)
        os.remove(model_file)
        os.remove(tar_path)
        os.rmdir(model_dir)

        # Deploy to SageMaker endpoint
        sagemaker_session = Session()
        pytorch_model = PyTorchModel(
            model_data=model_artifact_s3,
            role=get_execution_role(),
            framework_version="2.0",
            py_version="py310",
            entry_point="inference.py",
            source_dir=os.path.dirname(__file__),  # Use steps/ directory as source_dir
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

        return endpoint_name

    except Exception as e:
        logger.error(f"Error during deployment: {e}")
        raise e