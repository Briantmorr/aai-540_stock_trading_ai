from sagemaker.workflow.function_step import step
from steps.utils import get_default_bucket, upload_to_s3, setup_logging
import yfinance as yf
from datetime import datetime, timedelta


@step(instance_type="ml.m5.large", dependencies="requirements.txt")
def fetch(ticker, years):
    """
    Fetches historical stock data for the given ticker over the specified number of years,
    saves the raw data to S3, and returns the S3 path.
    """
    logger = setup_logging()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_date, end=end_date, interval="1d") 
        if stock_data.empty:
            logger.error(f"No data available for {ticker} from {start_date} to {end_date}.")
            return None
        logger.info(f"Successfully fetched data for {ticker}")
        file_name = f"{ticker}_raw_data.csv"
        csv_buffer = stock_data.to_csv()
        s3_path = upload_to_s3(csv_buffer, get_default_bucket(), f"raw/{file_name}")
        logger.info(f"Raw data saved to {s3_path}")
        return s3_path

    except Exception as e:
        logger.error(f"An error occurred while fetching data: {e}")
        raise e