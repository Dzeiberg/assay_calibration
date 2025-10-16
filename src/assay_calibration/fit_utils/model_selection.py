from src.assay_calibration.fit_utils.two_sample.fit import single_fit
from src.assay_calibration.data_utils.dataset import Scoreset,BasicScoreset
from tqdm.auto import trange
import json
from pathlib import Path
import numpy as np
import scipy.stats as sps
from typing import List, Dict
from joblib import Parallel, delayed,parallel_backend
import random
from src.assay_calibration.fit_utils.fit import makeOneHot, sample_specific_bootstrap,Fit
from src.assay_calibration.fit_utils.two_sample.density_utils import get_likelihood
from src.assay_calibration.fit_utils.utils import serialize_dict
import os
import pickle

def bootstrapped_likelihood_ratio_test(scoreset: Scoreset, N_bootstraps: int, save_dir, **kwargs):
    """
    Required Args:
    ------------------
    - scoreset : Scoreset : Scoreset to use in model selection
    - N_bootstraps : int : number of bootstrap iterations to run in the bootstrapped likelihood ratio test

    Optional kwargs:
    ------------------
    - N_restarts : int (default 100) : Number of restarts to in each bootstrap iteration
    - init_method : str in {'kmeans','method_of_moments'} (default 'kmeans'): what method to use in initializing the component parameters
    - cosntrained : bool (default True) : run the fits enforcing the monotonicity constraint
    - init_constraint_adjustment : str in {'skew','scale'} : which parameter to adjust to initially satisfy density constraints
    - save_filepath : Optional[str|Path] : file where the model selection results are saved
    """
    N_restarts = kwargs.get("N_restarts", 100)
    constrained = kwargs.get("constrained", True)
    init_method = kwargs.get("init_method", "random")
    if init_method == "random":
        init_method = ['kmeans','method_of_moments']
        random.shuffle(init_method)
        init_method = init_method[0]
    init_constraint_adjustment = kwargs.get("init_constraint_adjustment", "skew")
    scores = scoreset.scores
    sample_assignments = scoreset.sample_assignments
    sample_assignments = makeOneHot(sample_assignments)
    mask = sample_assignments.any(1) & (~np.isnan(scores))
    scores = scores[mask]
    sample_assignments = sample_assignments[mask]
    train_indices, val_indices = sample_specific_bootstrap(sample_assignments)
    scores_train, sample_assignments_train = (
        scores[train_indices],
        sample_assignments[train_indices],
    )
    scores_val, sample_assignments_val = (
        scores[val_indices],
        sample_assignments[val_indices],
    )
    model_two_comps = fit_iteration(
        scores_train,
        sample_assignments_train,
        2,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    model_three_comps = fit_iteration(
        scores_train,
        sample_assignments_train,
        2,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    likelihood_two_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        model_two_comps["component_params"],
        model_two_comps["weights"],
    )
    likelihood_three_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        model_three_comps["component_params"],
        model_three_comps["weights"],
    )
    observed_test_statistic = -2 * (likelihood_two_comp - likelihood_three_comp)
    sample_sizes = sample_assignments.sum(0)
    test_statistics = np.zeros(N_bootstraps)
    selection_results = {
        "observed_model_2_comp": model_two_comps,
        "observed_model_3_comp": model_three_comps,
        "observed_test_statistic": observed_test_statistic,
        "train_indices": train_indices,
        "val_indices": val_indices,
        'observed_likelihood_k2': likelihood_two_comp,
        'observed_likelihood_k3': likelihood_three_comp
    }
    # bootstrap_results = Parallel(n_jobs=min(N_bootstraps,10), verbose=N_bootstraps)(
    #     delayed(run_bootstrap_iter)(
    #         model_two_comps["component_params"],
    #         model_two_comps["weights"],
    #         sample_sizes,
    #         constrained,
    #         init_method,
    #         init_constraint_adjustment,
    #         N_restarts,
    #     )
    #     for _ in trange(N_bootstraps)
    # )
    save_dir = Path(save_dir)
    bootstrap_jobs = generate_bootstrap_iter_jobs(model_two_comps['component_params'],
                                                  model_two_comps['weights'],
                                                  sample_sizes,
                                                  constrained,
                                                    init_method,
                                                    init_constraint_adjustment,
                                                    N_bootstraps,
                                                    N_restarts,
                                                    save_dir/"bootstrap_fits",)
    
    jobs_dir = save_dir / 'jobs'
    jobs_dir.mkdir(exist_ok=True,parents=True)
    for job_num,job in enumerate(bootstrap_jobs):
        with open(jobs_dir / f"job_{job_num}.pkl",'wb') as f:
            pickle.dump(job,f)
    with open("model_selection_data.pkl",'wb') as f:
        pickle.dump(selection_results,f)
    return
    selection_results["bootstrap_results"] = bootstrap_results
    for i, result in enumerate(bootstrap_results):
        if result is None:
            raise ValueError(f"Bootstrap iteration {i} failed")
        test_statistics[i] = result["test_statistic"]
    p_value = (1 + (test_statistics >= observed_test_statistic).sum()) / (1 + N_bootstraps)
    selection_results["p_value"] = p_value
    selection_results["kwargs"] = kwargs
    save_filepath = kwargs.get("save_filepath", None)
    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        save_filepath.mkdir(parents=True, exist_ok=True)
        with open(save_filepath, "w") as f:
            json.dump(serialize_dict(selection_results), f)
        print(f"Model selection results written to {save_filepath}")
    print(f"Model selection p-value: {p_value}")
    return selection_results

def generate_bootstrap_iter_jobs(
    component_params,
    weights,
    sample_sizes,
    constrained,
    init_method,
    init_constraint_adjustment,
    N_bootstraps,
    N_restarts,
    save_dir,
):
    simulated_scores, simulated_sample_assignments = generate_scoreset(
        component_params, weights, sample_sizes
    )
    # train_indices, val_indices = sample_specific_bootstrap(simulated_sample_assignments)
    # trainScoreset = BasicScoreset(
    #     simulated_scores[train_indices],
    #     simulated_sample_assignments[train_indices],
    # )
    # valScoreset = BasicScoreset(
    #     simulated_scores[val_indices],
    #     simulated_sample_assignments[val_indices],
    # )
    fit = Fit(BasicScoreset(simulated_scores,simulated_sample_assignments)) # type: ignore
    jobs = []
    for bootstrapIter in range(N_bootstraps):
        fit_jobs = fit.generate_fit_jobs([2,3],save_dir,
                                        num_fits=N_restarts,
                                        score_min=simulated_scores.min(),
                                        score_max=simulated_scores.max(),
                                        bootstrap=True,
                                        bootstrap_seed=bootstrapIter)
        jobs+=fit_jobs
    return jobs

    bootstrap_model_k2 = fit_iteration(
        scores_train,
        sample_assignments_train,
        2,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    bootstrap_model_k3 = fit_iteration(
        scores_train,
        sample_assignments_train,
        3,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    likelihood_two_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        bootstrap_model_k2["component_params"],
        bootstrap_model_k2["weights"],
    )
    likelihood_three_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        bootstrap_model_k3["component_params"],
        bootstrap_model_k3["weights"],
    )

    bootstrap_result = {
        "test_statistic": -2 * (likelihood_two_comp - likelihood_three_comp),
        "bootstrapped_model_2_comp": bootstrap_model_k2,
        "bootstrapped_model_3_comp": bootstrap_model_k3,
        'likelihood_k2': likelihood_two_comp,
        'likelihood_k3': likelihood_three_comp
    }
    return serialize_dict(bootstrap_result)


def fit_iteration(
    scores,
    sample_assignments,
    n_components,
    constrained,
    init_method,
    init_constraint_adjustment,
    N_restarts,
)->Dict:
    # fits: List[Dict] = [
    #     single_fit(
    #         scores,
    #         sample_assignments,
    #         n_components,
    #         constrained,
    #         init_method,
    #         init_constraint_adjustment,
    #     )
    #     for _ in trange(N_restarts)
    # ]
    fits = list(Parallel(n_jobs=min(os.cpu_count() or 1,N_restarts), verbose=1)(
        delayed(single_fit)(
            scores,
            sample_assignments,
            n_components,
            constrained,
            init_method,
            init_constraint_adjustment,
        )
        for _ in range(N_restarts)
    ))
        
    # Sort fits by increasing likelihood
    fits.sort(key=lambda d: d["likelihoods"][-1])
    # Find iteration with best likelihood
    best_fit = fits[-1]
    return best_fit # type: ignore


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
