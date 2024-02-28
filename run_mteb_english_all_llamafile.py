"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""
import os
import subprocess
import logging
from mteb import MTEB
from rubra_model import RubraModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

from run_mteb_english import (
    TASK_LIST_CLASSIFICATION,
    TASK_LIST_CLUSTERING,
    TASK_LIST_PAIR_CLASSIFICATION,
    TASK_LIST_RERANKING,
    TASK_LIST_RETRIEVAL,
    TASK_LIST_STS,
)

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

# Function to run and then stop a model
def run_and_stop_model(model_file):
    model_name = model_file.replace(".llamafile", "")
    # Start the model file (set up the environment or load the model)
    process = subprocess.Popen(f"./{model_file}", shell=True)
    
    # Initialize your model here
    model = RubraModel()

    for task in TASK_LIST:
        logger.info(f"Running task: {task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Adjust for languages if necessary
        evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=eval_splits)

    # Attempt to terminate the subprocess
    process.terminate()
    try:
        process.wait(timeout=10)  # Wait up to 10 seconds for the process to terminate
    except subprocess.TimeoutExpired:
        logger.warning(f"Process did not terminate in time and will be killed forcefully.")
        process.kill()  # Forcefully kill the process if it doesn't terminate
        process.wait()  # Wait for the process to be killed

# List all model files in the current directory
model_files = [f for f in os.listdir('.') if f.endswith('.llamafile')]

# Iterate over each model file and run the evaluation
for model_file in model_files:
    logger.info(f"Evaluating model: {model_file}")
    run_and_stop_model(model_file)
