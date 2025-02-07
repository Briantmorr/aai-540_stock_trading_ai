import boto3
from sagemaker.workflow.function_step import step

@step(
    instance_type="ml.m5.large",
    dependencies="requirements.txt"
)
def preprocess(data_path):
    """
    Reads raw CSV data from S3, preprocesses it by resetting the index,
    formatting the date, selecting relevant features, sorting the data,
    and then saving the processed CSV back to S3. Returns the S3 path to the processed data.
    """
    try:
        # Setup S3 bucket
        sagemaker_session = Session()
        default_bucket = sagemaker_session.default_bucket()
        
        # Extract the S3 key from the full path.
        # Assumes the S3 path format is "s3://{default_bucket}/raw/<file_name>"
        bucket = default_bucket
        key = data_path.split(f"s3://{default_bucket}/")[1]
        
        # Read raw data from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw_data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        
        logging.info(f"Successfully read raw data from {data_path}")
        
        # Preprocessing steps:
        # 1. Reset index and format the 'Date' column.
        raw_data.reset_index(inplace=True)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'], utc=True).dt.strftime('%Y-%m-%d')
        
        # 2. Select only relevant columns
        features = ['Date', 'Open', 'High', 'Low', 'Close']
        processed_data = raw_data[features]
        
        # 3. Sort by date and reset index
        processed_data.sort_values('Date', inplace=True)
        processed_data.reset_index(drop=True, inplace=True)
        
        # Save processed data back to S3.
        # Change the S3 key from 'raw' to 'processed'
        processed_file_name = key.replace('raw', 'processed')
        csv_buffer = processed_data.to_csv(index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=processed_file_name,
            Body=csv_buffer
        )
        
        processed_data_path = f"s3://{bucket}/{processed_file_name}"
        logging.info(f"Preprocessed data saved to {processed_data_path}")
        return processed_data_path

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        raise e
