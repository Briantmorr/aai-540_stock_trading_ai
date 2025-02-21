import boto3
from sagemaker import Session
import logging

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