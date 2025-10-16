import sys
import numpy as np
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parents[1]))
import scipy.stats as sps
from src.assay_calibration.data_utils.dataset import Scoreset
import json
import argparse
from tqdm.auto import tqdm
from src.assay_calibration.fit_utils.utils import serialize_dict
from src.assay_calibration.fit_utils.model_selection import (
    bootstrapped_likelihood_ratio_test,
)


def generate_scoreset(params, weights, sample_sizes):
    samples = []
    sample_assignments = []
    assert weights.shape[1] == len(params)
    assert len(sample_sizes) == weights.shape[0]
    n_samples = len(sample_sizes)
    for sampleNum, (sample_weights, sample_size) in enumerate(
        zip(weights, sample_sizes)
    ):
        comp_sizes = np.round(sample_weights * sample_size).astype(int)
        for compParams, compSize in zip(params, comp_sizes):
            if compSize <= 0:
                continue
            samples.append(
                sps.skewnorm.rvs(
                    compParams[0], loc=compParams[1], scale=compParams[2], size=compSize
                )
            )
            sa = np.zeros((compSize, n_samples), dtype=bool)
            sa[:, sampleNum] = 1
            sample_assignments.append(sa)
    return np.concatenate(samples), np.concatenate(sample_assignments)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Model selection for scoresets.")
    # parser.add_argument("--scoreset_name", type=str, required=True, help="Name of the scoreset.")
    # parser.add_argument("--scoresets_dir", type=str, required=True, help="Directory containing the scoresets.")
    # parser.add_argument("--model_selection_save_dir", type=str, required=False, default=None, help="Directory to save model selection results.")
    # parser.add_argument("--n_bootstraps", type=int, required=False, default=100, help="Number of bootstrap runs to use in model selection")
    # parser.add_argument("--N_restarts", type=int, required=False, default=100, help="Number of bootstrap runs to use in model selection")
    # args = parser.parse_args()
    argdict = dict(
        scoreset_names=["TP53_Giacomelli_2018_p53null_etoposide",
                        "OTC_Lo_2023"],
        scoresets_dir="/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/scoresets/",
        model_selection_save_dir="/data/projects/igvf/assay_calibration/experiments/model_selection_testing/",
        pillar_project_filepath = "/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz",
        N_bootstraps=100,
        N_restarts=100,
    )
    args = argparse.Namespace(**argdict)
    df = pd.read_csv(args.pillar_project_filepath)
    scoreset_names = args.scoreset_names
    save_dir = Path(args.model_selection_save_dir)
    save_dir.mkdir(exist_ok=True,parents=True)
    for scoreset_name in tqdm(scoreset_names):
        scoreset = Scoreset(df[df.Dataset == scoreset_name])
        bootstrapped_likelihood_ratio_test(scoreset,args.N_bootstraps,
                                           save_dir / scoreset_name,
                                           N_restarts=args.N_restarts)