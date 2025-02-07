from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

import boto3
from sagemaker.workflow.function_step import step
from sagemaker import get_execution_role, Session
import logging

sagemaker_session = Session()
role = get_execution_role()

default_bucket = sagemaker_session.default_bucket()
session = PipelineSession(boto_session=sagemaker_session.boto_session, default_bucket=default_bucket)
pipeline_name = 'stock-pipeline'
default_ticker = "SPY"
years_of_data_to_fetch = 10

@step(
    instance_type="ml.m5.large",
    dependencies="requirements.txt"
)
def fetch(ticker=default_ticker, years=years_of_data_to_fetch):
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    import logging
    
    # Setup S3 bucket
    sagemaker_session = Session()
    default_bucket = sagemaker_session.default_bucket()
    
    # Define the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    try:
        # Fetch historical stock data
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval="1d")
        
        if stock_data.empty:
            logging.error(f"No data available for {ticker} from {start_date} to {end_date}.")
            return None
            
        logging.info(f"Successfully fetched data for {ticker}")
        
        # Save raw data to S3
        file_name = f"{ticker}_raw_data.csv"
        s3_path = f"s3://{default_bucket}/raw/{file_name}"
        
        # Save to S3 using boto3
        csv_buffer = stock_data.to_csv()
        boto3.client('s3').put_object(
            Bucket=default_bucket,
            Key=f"raw/{file_name}",
            Body=csv_buffer
        )
        
        logging.info(f"Raw data saved to {s3_path}")
        return s3_path
        
    except Exception as e:
        logging.error(f"An error occurred while fetching data: {e}")
        raise e


@step(
    instance_type="ml.m5.large",
    dependencies="requirements.txt"
)
def preprocess(data_path):
    import pandas as pd
    import logging
    from io import StringIO
    
    try:
        # Setup S3 bucket
        sagemaker_session = Session()
        default_bucket = sagemaker_session.default_bucket()
        
        # Extract bucket and key from s3 path
        bucket = default_bucket
        key = data_path.split(f"s3://{default_bucket}/")[1]
        
        # Read raw data from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw_data = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        
        logging.info(f"Successfully read raw data from {data_path}")
        
        # Preprocess steps
        # 1. Reset index and ensure date is properly formatted
        raw_data.reset_index(inplace=True)
        raw_data['Date'] = pd.to_datetime(raw_data['Date']).dt.strftime('%Y-%m-%d')
        
        # 2. Select only relevant columns
        features = ['Date', 'Open', 'High', 'Low', 'Close']
        processed_data = raw_data[features]
        
        # 3. Sort by date
        processed_data.sort_values('Date', inplace=True)
        processed_data.reset_index(drop=True, inplace=True)
        
        # Save processed data back to S3
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

@step(
    instance_type="ml.m5.large"
)
def train_and_eval(data):
    # retrieve preprocessed csv data
    # train model
    # load current model performance
    # if model is better than current
    # save model to s3 bucket /model with postfix of date
    # -- combining training and eval to prevent unecessary model saving  
    logging.info(f'data should be output of preprocess function: {data}')
    return "stub_train"

@step(
    instance_type="ml.t3.medium"
)
def deploy(data):
    # if latest_model is different than current
    # update deployment to use latest model
    logging.info(f'data should be output of train function: {data}')
    return "stub_eval"


fetch_result = fetch(default_ticker, years_of_data_to_fetch)
preprocess_result = preprocess(fetch_result)
train_and_eval_result = train_and_eval(preprocess_result)
deploy_result = deploy(train_and_eval_result)

pipeline = Pipeline(
    name=pipeline_name,
    steps=[fetch_result, preprocess_result, train_and_eval_result, deploy_result],
    sagemaker_session=session,
)

pipeline.upsert(role)