# preprocess.py
from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, upload_to_s3, setup_logging
import pandas as pd
from io import StringIO
import boto3


def preprocess_data(raw_data):
    """
    Preprocesses raw stock data by resetting the index, formatting dates,
    selecting relevant features, and sorting by date.
    Args:
        raw_data (pd.DataFrame): Raw stock data from S3.
    Returns:
        pd.DataFrame: Processed data with selected features and sorted dates.
    """
    # Reset index and format 'Date' column
    raw_data.reset_index(inplace=True)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], utc=True).dt.strftime('%Y-%m-%d')

    features = ['Date', 'Open', 'High', 'Low', 'Close']
    processed_data = raw_data[features]

    processed_data.sort_values('Date', inplace=True)
    processed_data.reset_index(drop=True, inplace=True)
    return processed_data


@step(instance_type="ml.m5.large", dependencies="requirements.txt")
def preprocess(data_path):
    """
    Reads raw CSV data directly from S3, preprocesses it, saves the processed CSV back to S3,
    and returns the S3 path to the processed data.
    Args:
        data_path (str): S3 path to the raw CSV file (e.g., 's3://bucket/raw/SPY_raw_data.csv').
    Returns:
        str: S3 path to the processed CSV file (e.g., 's3://bucket/processed/SPY_raw_data.csv').
    """
    logger = setup_logging()
    logger.info(f"Starting preprocessing with data from {data_path}")

    try:
        bucket = get_default_bucket()
        key = data_path.replace(f"s3://{bucket}/", "")
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw_data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        logger.info(f"Successfully read raw data from {data_path}")

        processed_data = preprocess_data(raw_data)
        logger.info("Data preprocessing completed")

        # Save processed data to S3
        processed_file_name = key.replace('raw', 'processed')
        processed_s3_key = f"processed/{processed_file_name}"
        csv_buffer = processed_data.to_csv(index=False)
        processed_data_path = upload_to_s3(csv_buffer, bucket, processed_s3_key)
        logger.info(f"Preprocessed data saved to {processed_data_path}")

        return processed_data_path

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e