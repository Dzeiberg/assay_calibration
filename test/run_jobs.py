from joblib import Parallel, delayed
import argparse
import pickle
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))
from src.assay_calibration.fit_utils.fit import Fit

def process_pickle_file(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            job_data = pickle.load(f)
        Fit.execute_fit_job(job_data)
        os.remove(pickle_path)
    except Exception as e:
        print(f"Error processing {pickle_path}: {e}")

def process_directory(directory):
    pickle_files = list(Path(directory).rglob("*.pkl"))
    Parallel(n_jobs=-1,
             verbose=100)(delayed(process_pickle_file)(str(pickle_file)) for pickle_file in pickle_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pickle files in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing pickle files.")
    args = parser.parse_args()

    process_directory(args.directory)