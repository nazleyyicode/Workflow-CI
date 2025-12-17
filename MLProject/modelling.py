import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# load dataset preprocessing
df = pd.read_csv("namadataset_preprocessing/weather_forecast_preprocessing.csv")

# menentukan fitur dan target
X = df.drop(columns=["Rain"])
y = df["Rain"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    # autologging
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Akurasi :", accuracy_score(y_test, y_pred))

    # menyimpan un ID ke file txt agar bisa dibaca oleh GitHub Actions
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)