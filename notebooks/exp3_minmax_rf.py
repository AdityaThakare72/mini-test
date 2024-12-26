import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import dagshub
import os
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("https://raw.githubusercontent.com/michaelmml/Financial-Data-Science/refs/heads/main/loan_approval_dataset.csv")


df.columns = df.columns.str.replace(' ', '')

x = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

x = pd.get_dummies(x)

scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)



params_grid = {
    "n_estimators": [15, 20, 30],
    "max_depth": [10, 20],
    }

dagshub.init(repo_owner='AdityaThakare72', repo_name='mini-test', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/AdityaThakare72/mini-test.mlflow")
mlflow.set_experiment("exp3_minmax_rf")


# start the parent run for hp tuning
with mlflow.start_run(run_name="exp3_minmax_rf") as parent_run:
    
    grid_search = GridSearchCV(RandomForestClassifier(), params_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # log each parameter comb as child run
    for params in grid_search.cv_results_['params']:
        with mlflow.start_run(run_name=f"exp3_minmax_rf_{params}", nested=True) as child_run:
            model = RandomForestClassifier(**params)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, pos_label=' Approved')
            precision = precision_score(y_test, y_pred, pos_label=' Approved')

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)

            mlflow.log_params(params)

            # print the results
            print(f"Model: {params}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}")

        # log the best model in parent run
        mlflow.log_param("best_model", grid_search.best_params_)
        mlflow.log_metric("best_f1_score", grid_search.best_score_)

        print(f"Best Model: {grid_search.best_params_}, Best F1 Score: {grid_search.best_score_}")

        # save and log the code
        mlflow.log_artifact(__file__)

        # log the model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

