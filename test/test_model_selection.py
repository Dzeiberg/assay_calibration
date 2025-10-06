import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import scipy.stats as sps
from src.assay_calibration.data_utils.dataset import Scoreset
import json
import argparse
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
        scoreset_name="BRCA1_Findlay_2018",
        scoresets_dir="/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/scoresets/",
        model_selection_save_dir="test/model_selection_test_results/",
        N_bootstraps=25,
        N_restarts=25,
    )
    args = argparse.Namespace(**argdict)

    scoreset_name = args.scoreset_name
    scoresets_dir = Path(args.scoresets_dir)
    if not scoresets_dir.exists():
        raise ValueError(f"scoresets dir: {scoresets_dir} does not exist")
    scoreset_filepath = scoresets_dir / f"{scoreset_name}.json"
    if not scoreset_filepath.is_file():
        raise ValueError()
    scoreset = Scoreset.from_json(scoreset_filepath)
    res = bootstrapped_likelihood_ratio_test(scoreset, 10, N_restarts=args.N_restarts)
    save_dir = Path(args.model_selection_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"test_result_{scoreset_name}.json", "w") as f:
        json.dump(res, f, indent=4)
