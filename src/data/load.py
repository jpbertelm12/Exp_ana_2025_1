from sklearn.datasets import load_wine  # <-- Cambio aquí
from sklearn.model_selection import train_test_split 
import pandas as pd 
import argparse
import wandb 
import numpy as np
###
# 🎯 Argument parser for execution ID.
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=0.8, val_size=0.1, random_state=42):
    """
    Loads and splits the Wine dataset into train, validation, and test sets.
    """
    wine = load_wine()  # <-- Cambio aquí
    X = wine.data
    y = wine.target

    # Split into training and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )

    # Split temp into validation and test
    val_ratio = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_and_log():
    """
    Loads the data, formats it as DataFrames, and logs it to Weights & Biases (W&B)
    as a versioned artifact.
    """
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Load Raw Data ExecId-{args.IdExecution}",
        job_type="load-data"
    ) as run:

        # Load dataset splits
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load()

        # Convert to DataFrames
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df["target"] = y_train

        val_df = pd.DataFrame(X_val, columns=feature_names)
        val_df["target"] = y_val

        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df["target"] = y_test

        datasets = [train_df, val_df, test_df]
        names = ["training", "validation", "test"]

        # Create and populate the artifact
        raw_data = wandb.Artifact(
            name="wine-raw",  # <-- Cambio aquí
            type="dataset",
            description="Wine dataset split into train/val/test",  # <-- Cambio aquí
            metadata={
                "source": "sklearn.datasets.load_wine",  # <-- Cambio aquí
                "sizes": [len(df) for df in datasets]
            }
        )

        for name, df in zip(names, datasets):
            with raw_data.new_file(f"{name}.csv", mode="w") as f:
                df.to_csv(f, index=False)

        run.log_artifact(raw_data)

# Run the function.
load_and_log()
