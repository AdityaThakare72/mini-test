import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import dagshub
import os

df = pd.read_csv("https://raw.githubusercontent.com/michaelmml/Financial-Data-Science/refs/heads/main/loan_approval_dataset.csv")

df.columns = df.columns.str.replace(' ', '')

x = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

x = pd.get_dummies(x)

scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}

models = {"RandomForestClassifier": RandomForestClassifier(), 
          "LogisticRegression": LogisticRegression(), 
          "SVC": SVC()}


# set the experiment name and tracking uri
dagshub.init(repo_owner='AdityaThakare72', repo_name='mini-test', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mini-test.mlflow")

mlflow.set_experiment("exp1_standardvsminmax")

# start the parent run 
with mlflow.start_run(run_name="exp1_standardvsminmax") as parent_run:
    # loop through the scalers and models
    for algo, model in models.items():
        for scaler_name, scaler in scalers.items():
            # start the child run
            with mlflow.start_run(run_name=f"{algo}_{scaler_name}", nested=True) as child_run:
                x_scaled = scaler.fit_transform(x)

                x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
                model.fit(x_train, y_train)

                mlflow.log_param("scaler", scaler_name)
                mlflow.log_param("model", algo)
                mlflow.log_param("test_size", 0.2)
                

                y_pred = model.predict(x_test)

                # log the metrics
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, pos_label=' Approved')
                precision = precision_score(y_test, y_pred, pos_label=' Approved')

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("precision", precision)
                mlflow.sklearn.log_model(model, "model")
               
                # log the model
                mlflow.sklearn.log_model(model, "model")

                # print results
                print(f"Model: {algo}, Scaler: {scaler_name}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}")












