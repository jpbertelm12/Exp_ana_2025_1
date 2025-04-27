# Import the model class
from sklearn.ensemble import RandomForestClassifier  # <-- Cambio aquí
import os
import argparse
import wandb
import pickle
#
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    os.makedirs("./model")

# Data parameters testing (en Wine son 13 features)
input_shape = 13  # <-- Cambio aquí (iris tenía 4, wine tiene 13)

# Define the model filename
model_filename = "random_forest_classifier_model.pkl"  # <-- Cambio aquí

# Función para construir y loguear modelo
def build_model_and_log(config, model, model_name="Random Forest Classifier", model_description="Random Forest Classifier for Wine Dataset"):
    with wandb.init(
        project="MLOps-Pycon2023", 
        name=f"Initialize Model ExecId-{args.IdExecution}", 
        job_type="initialize-model",
        config=config
    ) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name,
            type="model",
            description=model_description,
            metadata=dict(config)
        )

        # Save the trained model
        with open(f"./model/{model_filename}", "wb") as f:
            pickle.dump(model, f)

        # Add the model to the artifact
        model_artifact.add_file(f"./model/{model_filename}")

        # Save and log the artifact
        wandb.save(f"./model/{model_filename}")
        run.log_artifact(model_artifact)

# Model configuration
model_config = {
    "input_shape": input_shape,
    "n_estimators": 100,     # <-- Estimadores típicos para Random Forest
    "max_depth": None,       # <-- Sin límite de profundidad
    "random_state": 42       # <-- Semilla para reproducibilidad
}

# Create the model
model = RandomForestClassifier(
    n_estimators=model_config["n_estimators"],
    max_depth=model_config["max_depth"],
    random_state=model_config["random_state"]
)

# Log the model and configuration
build_model_and_log(
    model_config,
    model,
    model_name="random_forest_classifier",
    model_description="Random Forest Classifier for Wine Dataset"
)
