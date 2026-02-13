import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from prefect import task, flow

import joblib
import os


# =====================================
# TASK 1: Load Data
# =====================================

@task
def load_data(file_path):

    return pd.read_csv(file_path)


# =====================================
# TASK 2: Prepare Inputs & Output
# =====================================

@task
def prepare_data(data, text_col, rating_col):

    data = data[[text_col, rating_col]]
    data.dropna(inplace=True)

    # Create Sentiment
    def make_sentiment(r):

        if r >= 4:
            return 1
        elif r <= 2:
            return 0
        else:
            return None

    data["sentiment"] = data[rating_col].apply(make_sentiment)

    data.dropna(inplace=True)

    X = data[text_col]
    y = data["sentiment"]

    return X, y


# =====================================
# TASK 3: Train-Test Split
# =====================================

@task
def split_data(X, y, test_size=0.3):

    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )


# =====================================
# TASK 4: Vectorize Text
# =====================================

@task
def vectorize_text(X_train, X_test, max_features):

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    return X_train_vec, X_test_vec, tfidf


# =====================================
# TASK 5: Train Model
# =====================================

@task
def train_model(X_train_vec, y_train, hyperparameters):

    model = LogisticRegression(
        C=hyperparameters["C"],
        solver=hyperparameters["solver"],
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    return model


# =====================================
# TASK 6: Evaluate Model
# =====================================

@task
def evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test):

    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    return train_acc, test_acc, f1


# =====================================
# TASK 7: Log to MLflow
# =====================================

@task
def log_to_mlflow(model, tfidf, metrics, params):

    train_acc, test_acc, f1 = metrics

    mlflow.set_experiment("FLIPKART_PREFECT_RUNS")

    with mlflow.start_run():

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("f1_score", f1)

        # Save model
        model_path = "final_model.pkl"
        joblib.dump((tfidf, model), model_path)

        size = os.path.getsize(model_path)

        mlflow.log_metric("model_size", size)

        mlflow.log_artifact(model_path)

        mlflow.sklearn.log_model(model, "model")

        os.remove(model_path)


# =====================================
# FLOW (MAIN WORKFLOW)
# =====================================

@flow(name="Flipkart Sentiment Training Flow")
def workflow():

    # ---------------- Config ----------------

    DATA_PATH = r"C:\Users\neell\Downloads\reviews_data_dump\reviews_badminton\data.csv"

    TEXT_COL = "Review text"     
    RATING_COL = "Ratings"        

    HYPERPARAMETERS = {
        "C": 1.0,
        "solver": "liblinear",
        "max_features": 10000
    }

    # ---------------- Pipeline ----------------

    # Load
    data = load_data(DATA_PATH)

    # Prepare
    X, y = prepare_data(data, TEXT_COL, RATING_COL)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Vectorize
    X_train_vec, X_test_vec, tfidf = vectorize_text(
        X_train,
        X_test,
        HYPERPARAMETERS["max_features"]
    )

    # Train
    model = train_model(
        X_train_vec,
        y_train,
        HYPERPARAMETERS
    )

    # Evaluate
    train_acc, test_acc, f1 = evaluate_model(
        model,
        X_train_vec,
        y_train,
        X_test_vec,
        y_test
    )

    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("F1 Score:", f1)

    # MLflow Logging
    log_to_mlflow(
        model,
        tfidf,
        (train_acc, test_acc, f1),
        HYPERPARAMETERS
    )


# =====================================
# DEPLOYMENT
# =====================================

if __name__ == "__main__":

    workflow.serve(
        name="flipkart-auto-training",
        cron="*/5 * * * * " 
    )
