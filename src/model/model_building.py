import numpy as np
import pandas as pd
import pickle
import yaml
import mlflow
from mlflow.sklearn import log_model

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dagshub
dagshub.init(repo_owner='CodeNeuron58', repo_name='CC-Emotion-Detection', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/CodeNeuron58/CC-Emotion-Detection.mlflow")
mlflow.set_experiment("emotion_detection_experiment")

# Custom logger
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import get_logger
logger = get_logger("model_building")


def load_data(train_data_path):
    logger.info(f"Loading training data from {train_data_path}")
    try:
        train_data = pd.read_csv(train_data_path)
        logger.info(f"Training data loaded successfully with shape {train_data.shape}")
        return train_data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def split_data(train_data):
    logger.info("Splitting training data into features and labels...")
    try:
        X = train_data.iloc[:, :-1].values
        y = train_data.iloc[:, -1].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Split completed. X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    except Exception as e:
        logger.error(f"Error splitting training data: {e}")
        raise


def model_building(X_train, y_train, X_val, y_val, params_file="params.yaml"):
    logger.info("Building GradientBoostingClassifier model...")
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)["model_building"]

        n_estimators = params["n_estimators"]
        learning_rate = params["learning_rate"]

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )

        with mlflow.start_run():
            logger.info(f"Training model with parameters: n_estimators={n_estimators}, "
                        f"learning_rate={learning_rate}")
            
            clf.fit(X_train, y_train)

            # Predictions for validation
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            # Log params and metrics
            mlflow.log_param("n_estimators",n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_metric("accuracy", float(acc))  # Convert accuracy to float for loggingacc)

 
            logger.info(f"Model training completed successfully with validation accuracy={acc:.4f}")

        return clf
    except Exception as e:
        logger.error(f"Error in model building: {e}")
        raise


def save_model(clf, model_path="models/model.pkl"):
    logger.info(f"Saving trained model to {model_path}")
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(clf, file)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    logger.info("Starting model building pipeline...")
    train_data = load_data('data/processed/train.csv')
    X_train, X_val, y_train, y_val = split_data(train_data)
    clf = model_building(X_train, y_train, X_val, y_val)
    save_model(clf)  # local pickle save
    logger.info("Model building pipeline completed successfully.")


if __name__ == '__main__':
    main()
