"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""
import os
import subprocess
import logging
import time
from mteb import MTEB
from rubra_model import RubraModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

MODEL_FILES_DIRECTORY = "./path/to/llamafiles"  # Update this path as needed

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

# Function to set executable permissions, run the model, wait, and then stop the model
def chmod_and_run_model(model_file):
    # Assuming model_file is the absolute path to the .llamafile
    model_name = os.path.basename(model_file).replace(".llamafile", "")
    
    # Set the file to be executable
    os.chmod(model_file, 0o755)  # Sets the file to be executable by the owner

    # Start the model file (set up the environment or load the model)
    # Directly use the absolute path without './' prefix
    process = subprocess.Popen(model_file, shell=True)
    
    logger.info(f"Waiting for 30 seconds for the model to load: {model_file}")
    time.sleep(30)  # Pause execution for 30 seconds

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
model_files = [f for f in os.listdir(MODEL_FILES_DIRECTORY) if f.endswith('.llamafile')]

# Iterate over each model file and run the evaluation
for model_file in model_files:
    full_path_model_file = os.path.join(MODEL_FILES_DIRECTORY, model_file)  # Generate full path
    logger.info(f"Evaluating model: {full_path_model_file}")
    chmod_and_run_model(full_path_model_file)
