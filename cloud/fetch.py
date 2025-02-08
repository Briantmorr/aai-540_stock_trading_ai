from sagemaker.workflow.function_step import step

@step(
    instance_type="ml.m5.large",
    dependencies="requirements.txt"
)
def fetch(ticker, years):
    """
    Fetches historical stock data for the given ticker over the specified number of years,
    saves the raw data to S3, and returns the S3 path.
    """
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    import logging
    from sagemaker import Session
    import boto3

    sagemaker_session = Session()
    default_bucket = sagemaker_session.default_bucket()
    
    # Define the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval="1d")
        
        if stock_data.empty:
            logging.error(f"No data available for {ticker} from {start_date} to {end_date}.")
            return None
        
        logging.info(f"Successfully fetched data for {ticker}")
        
        file_name = f"{ticker}_raw_data.csv"
        s3_path = f"s3://{default_bucket}/raw/{file_name}"
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