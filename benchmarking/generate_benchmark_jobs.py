import pandas as pd

from argparse import Namespace
from tqdm.auto import tqdm, trange
from pathlib import Path
import pickle
import argparse
import sys
from typing import Dict,Tuple
sys.path.append(str(Path(__file__).parent.parent))
from src.assay_calibration.data_utils.dataset import Scoreset
from src.assay_calibration.fit_utils.fit import Fit


def load_dataframe(dataframe_filepath):
    df = pd.read_csv(dataframe_filepath)
    return df

def initialize_scoreset_params():
    scoreset_params = {
    'clinvar_2018_0star': Namespace(clinvar_release='2018',
                                    min_clinvar_star=0),
    'clinvar_2018_2star': Namespace(clinvar_release='2018',
                                    min_clinvar_star=2),
    'clinvar_2025_0star': Namespace(clinvar_release='2025',
                                    min_clinvar_star=0),
    'clinvar_2025_2star': Namespace(clinvar_release='2025',
                                    min_clinvar_star=2)}
    scoreset_args = [("BRCA1_Findlay_2018",list(scoreset_params.keys())),
                    ("MSH2_Jia_2021", list(scoreset_params.keys())),
                    ("VHL_Buckley_2024", ['clinvar_2025_0star','clinvar_2025_2star']),
                    ('FKRP_Ma_2024',['clinvar_2025_0star',]),
                    ("LARGE1_Ma_2024",['clinvar_2025_0star'])]
    return scoreset_params, scoreset_args

def generate_scoresets(df, scoreset_params, scoreset_args)->Dict[Tuple[str,str],Scoreset]:
    scoresets = {}
    for scoreset_name, scoreset_paramsets in tqdm(scoreset_args):
        scoreset_df = df[df.Dataset == scoreset_name]
        for paramset_name in tqdm(scoreset_paramsets,leave=False):
            scoreset = Scoreset(scoreset_df,**scoreset_params[paramset_name].__dict__)
            scoresets[(scoreset_name,paramset_name)] = scoreset
    return scoresets

def generate_jobs(scoresets, fits_save_rt, jobs_save_rt, **kwargs):
    NBootstraps = kwargs.get("NBootstraps",1000)
    fits_save_rt = Path(fits_save_rt)
    jobs_save_rt = Path(jobs_save_rt)
    total_jobs = 0
    for (scoreset_name,paramset_name), scoreset in tqdm(list(scoresets.items())):
        fit = Fit(scoreset)
        uid = "_".join((scoreset_name,paramset_name))
        jobs_save_dir = jobs_save_rt / uid
        jobs_save_dir.mkdir(exist_ok=True,parents=True)
        for bootstrap_seed in trange(NBootstraps,leave=False):
            scoreset_jobs = fit.generate_fit_jobs([2,3],
                                                fits_save_rt / "_".join((scoreset_name,paramset_name)),
                                                bootstrap_seed=bootstrap_seed)
            for jobNum,job in enumerate(scoreset_jobs):
                with open(jobs_save_dir / f"job_{job['job_id']}.pkl",'wb') as f:
                    pickle.dump(job,f)
                total_jobs+=1
    print(f"Wrote {total_jobs:,d} jobs to {jobs_save_rt}")

def main(dataframe_filepath, fits_save_rt, jobs_save_rt):
    df = load_dataframe(dataframe_filepath)
    scoreset_params, scoreset_args = initialize_scoreset_params()
    scoresets = generate_scoresets(df, scoreset_params, scoreset_args)
    generate_jobs(scoresets, fits_save_rt,jobs_save_rt)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate benchmark jobs for assay calibration.")
    parser.add_argument("dataframe_filepath", type=str, help="Path to the input dataframe CSV file.")
    parser.add_argument("fits_save_rt", type=str, help="Root directory to save fit results.")
    parser.add_argument("jobs_save_rt", type=str, help="Root directory to save generated jobs.")
    parser.add_argument("--NBootstraps", type=int, default=1000, help="Number of bootstraps to generate (default: 1000).")

    args = parser.parse_args()

    main(args.dataframe_filepath, args.fits_save_rt, args.jobs_save_rt)