import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mini-test.mlflow")

dagshub.init(repo_owner='AdityaThakare72', repo_name='mini-test', mlflow=True)

import mlflow

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
