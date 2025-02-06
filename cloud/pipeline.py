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
    # for default ticker (SPY), 
    # fetch latest data from yfinance
    # Save to default_bucket
    logging.info('executing fetch step')
    return "stub_fetch"


@step(
    instance_type="ml.m5.large"
)
def preprocess(data):
    # retrieve latest data csv
    # remove uncessary columns, format rows, etc
    # save to bucket with postfix _preprocessed
    logging.info(f'data should be output of fetch function: {data}')
    return "stub_preprocess"

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
    instance_type="ml.m5.large"
)
def deploy(data):
    # if latest_model is different than current
    # update deployment to use latest model
    logging.info(f'data should be output of train function: {data}')
    return "stub_eval"


fetch_result = fetch()
preprocess_result = preprocess(fetch_result)
train_and_eval_result = train_and_eval(preprocess_result)
deploy_result = deploy(train_and_eval_result)

pipeline = Pipeline(
    name=pipeline_name,
    steps=[fetch_result, preprocess_result, train_and_eval_result, deploy_result],
    sagemaker_session=session,
)

pipeline.upsert(role)