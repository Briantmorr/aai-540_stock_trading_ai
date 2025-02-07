# Steps to implement

1. Navigate to your sagemaker domain
2. Open your jupyter lab
3. Open a terminal
4. Clone the repository
5. run ``` pip install yfinance pandas```
6. run the initial_setup notebook to create the s3 bucket
7. run ```python3 pipeline.py```
8. navigate to sagemakerUI -> Pipelines -> stock-pipeline and manually execute
9. verify execution by visiting cloudwatch logs > /aws/sagemaker/TrainingJobs and search for logger.info statements

> in later stages we will run this pipeline with pipeline.start() to automate execution