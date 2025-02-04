from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

import boto3
from sagemaker.workflow.function_step import step
from sagemaker import get_execution_role, Session

sagemaker_session = Session()
role = get_execution_role()

default_bucket = sagemaker_session.default_bucket()
session = PipelineSession(boto_session=sagemaker_session.boto_session, default_bucket=default_bucket)
pipeline_name = 'stock-pipeline'


@step
def fetch():
    # fetch data
    ticker = "SPY"
    print('hello world')
    return ticker

@step
def preprocess(data):
    # fetch data
    ticker = "SPY"
    print(f"Data should be output of fetch step: {data}")
    return ticker

step1_result = fetch()
step2_result = preprocess(step1_result)

pipeline = Pipeline(
    name=pipeline_name,
    steps=[fetch, preprocess],
    sagemaker_session=session,
)


pipeline.create(role)