import pandas as pd
import os
import argparse
import wandb
from sklearn.ensemble import RandomForestClassifier  # <-- Cambio aquí
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
#
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Read CSV data
def read_csv_data(data_dir, split):
    filepath = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(filepath)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    return X, y

# Train model
def train_model(X_train, y_train, config):
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return acc, f1, precision, recall, cm, report

# Main training and evaluation function
def train_and_evaluate(config, experiment_id='99'):
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Train-Eval RandomForest ExecId-{args.IdExecution} ExperimentId-{experiment_id}",
        job_type="train-eval", config=config
    ) as run:

        data_artifact = run.use_artifact('wine-preprocessed:latest')  # <-- Cambio aquí
        data_dir = data_artifact.download()

        X_train, y_train = read_csv_data(data_dir, "training")
        X_val, y_val = read_csv_data(data_dir, "validation")
        X_test, y_test = read_csv_data(data_dir, "test")

        # Train
        model = train_model(X_train, y_train, config)

        # Evaluate on validation
        val_acc, val_f1, val_precision, val_recall, _, _ = evaluate_model(model, X_val, y_val)
        print(f"Validation Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Evaluate on test
        test_acc, test_f1, test_precision, test_recall, confusion_mtx, full_report = evaluate_model(model, X_test, y_test)
        print(f"Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")

        # Log metrics
        wandb.log({
            "validation/accuracy": val_acc,
            "validation/f1": val_f1,
            "validation/precision": val_precision,
            "validation/recall": val_recall,
            "test/accuracy": test_acc,
            "test/f1": test_f1,
            "test/precision": test_precision,
            "test/recall": test_recall,
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=model.predict(X_test),
                y_true=y_test,
                class_names=[str(i) for i in sorted(set(y_test))]
            )
        })

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--IdExecution', type=str, help='ID of the execution')
    args = parser.parse_args()

    if not args.IdExecution:
        args.IdExecution = "testing-console"

    # You can tune these parameters
    experiments = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 20}
    ]

    for id, config in enumerate(experiments):
        train_and_evaluate(config, experiment_id=id)
