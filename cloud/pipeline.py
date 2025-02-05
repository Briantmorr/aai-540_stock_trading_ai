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


@step(
    instance_type="ml.m5.large"
)
def fetch():
    # fetch data
    ticker = "SPY"
    logging.info('executing fetch step')
    return ticker


@step(
    instance_type="ml.m5.large"
)
def preprocess(data):
    # fetch data
    ticker = "SPY"
    logging.info(f'data should be output of fetch function: {data}')
    return ticker


fetch_result = fetch()
preprocess_result = preprocess(fetch_result)

pipeline = Pipeline(
    name=pipeline_name,
    steps=[fetch_result, preprocess_result],
    sagemaker_session=session,
)

pipeline.upsert(role)