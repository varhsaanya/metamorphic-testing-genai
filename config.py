import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JM1_URL = (
    "https://raw.githubusercontent.com/klainfo/defect-datasets/"
    "master/JM1/JM1.csv"
)
JM1_LOCAL = DATA_DIR / "jm1.csv"

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
TARGET_COLUMN  = "defects"          

MODELS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "random_state": RANDOM_STATE,
    },
    "gradient_boosting": {
        "n_estimators": 150,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
    },
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
}
DEFAULT_MODEL = "random_forest"

MR_NOISE_SCALE       = 0.01   
MR_SCALE_FACTOR      = 1.10  
MR_PERMUTATION_SEED  = 0      
N_RANDOM_TEST_RUNS   = 500    

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
MR_CACHE_FILE  = RESULTS_DIR / "genai_mr_cache.json"

VIOLATION_THRESHOLD = 0.50   
