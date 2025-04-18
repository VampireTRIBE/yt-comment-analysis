import mlflow
import random

mlflow.set_tracking_uri("http://34.205.16.143:5000/")
with mlflow.start_run():
   mlflow.log_param("para1", random.randint(1,100))
   mlflow.log_param("para2", random.random())
   
   mlflow.log_metric("metric1", random.random())
   mlflow.log_metric("metric1", random.uniform(0.5,1.5))
   print("done")