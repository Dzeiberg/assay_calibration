import pickle
from src.assay_calibration.fit_utils.fit import Fit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a job from a specified file path.")
    parser.add_argument("job_filepath", type=str, help="Path to the job file.")
    args = parser.parse_args()
    job_filepath = args.job_filepath
    with open(job_filepath,'rb') as f:
        job = pickle.load(f)
    Fit.execute_fit_job(job)