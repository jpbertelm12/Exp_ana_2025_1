import pandas as pd 
import os
import argparse
import wandb
from sklearn.preprocessing import StandardScaler
#
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing-console"

# Preprocess function (normalize features)
def preprocess(X, y, normalize=True):
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, y

# Read CSV and return features/target
def read(data_dir, split):
    filename = split + ".csv"
    df = pd.read_csv(os.path.join(data_dir, filename))

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    return X, y

# Main pipeline: read, preprocess, and log back
def preprocess_and_log(steps):

    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Preprocess Data ExecId-{args.IdExecution}",
        job_type="preprocess-data"
    ) as run:    

        processed_data = wandb.Artifact(
            "wine-preprocessed",  # <-- Cambio aquí
            type="dataset",
            description="Preprocessed WINE dataset (normalized)",  # <-- Cambio aquí
            metadata=steps
        )
         
        # 🧊 Load raw artifact
        raw_data_artifact = run.use_artifact('wine-raw:latest')  # <-- Cambio aquí
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")

        for split in ["training", "validation", "test"]:
            X, y = read(raw_dataset, split)
            X_processed, y_processed = preprocess(X, y, **steps)

            # Rebuild DataFrame and save as CSV
            df = pd.DataFrame(X_processed, columns=[f"feature_{i}" for i in range(X_processed.shape[1])])
            df["target"] = y_processed

            with processed_data.new_file(f"{split}.csv", mode="w") as file:
                df.to_csv(file, index=False)

        # Upload the new artifact
        run.log_artifact(processed_data)

# Trigger the pipeline
steps = {"normalize": True}
preprocess_and_log(steps)
