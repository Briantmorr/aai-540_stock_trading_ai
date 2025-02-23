from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, setup_logging
import boto3
import sagemaker
import logging
from datetime import datetime

@step(instance_type="ml.m5.large")
def deploy(eval_output):
    """
    Approves and deploys the model from the Model Registry if deploy_flag is True, updates the current model in S3,
    and returns the endpoint name or None.
    Args:
        eval_output (tuple): Tuple (deploy_flag: bool, model_package_arn: str) from eval step.
    Returns:
        str or None: SageMaker endpoint name if deployed, else None.
    """
    logger = setup_logging()
    deploy_flag, model_package_arn = eval_output  # Unpack only two values
    logger.info(f"Received eval output: deploy_flag={deploy_flag}, model_package_arn={model_package_arn}")

    try:
        if not deploy_flag:
            logger.info("Deploy flag is False; skipping deployment.")
            return None

        sagemaker_session = sagemaker.Session()
        sm_client = boto3.client("sagemaker")

        sm_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved"
        )
        logger.info(f"Model package {model_package_arn} approved.")

        model = sagemaker.model.ModelPackage(
            model_package_arn=model_package_arn,
            role=sagemaker.get_execution_role(),
            sagemaker_session=sagemaker_session
        )
        endpoint_name = f"stock-pipeline-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large",
            endpoint_name=endpoint_name
        )
        logger.info(f"Model deployed to endpoint: {endpoint_name}")

        bucket = get_default_bucket()
        current_model_path = f"s3://{bucket}/model/current_model.safetensors"
        describe_response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
        model_s3_uri = describe_response["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        s3_client = boto3.client('s3')
        s3_client.copy_object(
            Bucket=bucket,
            Key="model/current_model.safetensors",
            CopySource={'Bucket': bucket, 'Key': model_s3_uri.split('/', 3)[-1]}
        )
        logger.info(f"Updated current model at {current_model_path}")

        return endpoint_name

    except Exception as e:
        logger.error(f"Error during deployment: {e}")
        raise e