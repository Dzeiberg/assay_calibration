import sys
import os
import math
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
logging.getLogger('matplotlib').setLevel(logging.ERROR)
sys.path.append(str(Path(__file__).parents[1]))
from src.assay_calibration.fit_utils.fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from src.assay_calibration.fit_utils.two_sample import density_utils  # noqa: E402
from src.assay_calibration.data_utils.dataset import Scoreset  # noqa: E402
from src.assay_calibration.fit_utils.utils import serialize_dict  # noqa: E402



def summarize_scoreset(fits,scoreset,save_filepath, **kwargs):
    """
    Summarizes a scoreset based on the provided arguments.
    Args:
        args: An object containing the following attributes:
            - fits (List[Dict]) : List of fit results
            - scoreset_name (str) : Name of the scoreset to summarize
            - df (pandas.DataFrame, optional): A pandas DataFrame to be summarized. 
              If provided, this will be used directly.
            - pillar_df_filepath (str, optional): A file path to a CSV file. If `df` 
              is not provided, the CSV file at this path will be read into a pandas 
              DataFrame and used for summarization.
    Optional Keyword Args:
        - point_values (List[int], optional): List of point values to assign. Defaults to 
          [1,2,3,4,5,6,7,8].
        - pathogenic_idx (int, optional): Index of the pathogenic component. Defaults to 0.
        - benign_idx (int, optional): Index of the benign component. Defaults to 1.
        - tolerance (float, optional): Tolerance for convergence in prior estimation. Defaults to 1e-4.
        - max_em_steps (int, optional): Maximum number of EM steps for prior estimation. Defaults to 10000.

    Note:
        Either `df` or `pillar_df_filepath` must be provided in `args`. If both are 
        provided, `df` takes precedence.
    """
    priors, point_ranges, score_range, log_fp, log_fb = process_fits(fits,scoreset,)
    results = dict(prior=np.nanmedian(priors),
                   point_ranges=point_ranges,
                   priors=priors,
                   score_range=score_range,
                   log_lr_plus=log_fp - log_fb)
    results = serialize_dict(results)
    save_filepath = Path(save_filepath)
    save_filepath.parent.mkdir(exist_ok=True,parents=True)
    with open(save_filepath,'w') as f:
        json.dump(results,f,indent=2)
    save_filepath_compact = save_filepath.parent / f"{save_filepath.stem}_compact.json"
    with open(save_filepath_compact,'w') as f:
        json.dump({k: results[k] for k in ['prior','point_ranges']},
                  f,indent=2)
    scoreset_fit_figure = plot_scoreset(scoreset, results, fits,score_range)
    figure_filepath = save_filepath.parent / f"{save_filepath.stem}figure_fits.png"
    scoreset_fit_figure.savefig(figure_filepath,bbox_inches='tight',dpi=300)
    plt.close(scoreset_fit_figure) 
    summary_fig = plot_summary(scoreset, fits, results, score_range, log_fp, log_fb)
    summary_figure_filepath = save_filepath.parent / f"{save_filepath.stem}_figure_summary.png"
    summary_fig.savefig(summary_figure_filepath,bbox_inches='tight',dpi=300)
    plt.close(summary_fig)

def process_fits(fits, scoreset,**kwargs)->Tuple[np.ndarray,Dict[int,List[Tuple[float,float]]],np.ndarray,np.ndarray,np.ndarray]:
    n_cores = os.cpu_count() or 1
    fit_priors = np.array(Parallel(n_jobs=min(len(fits), n_cores), verbose=10)(delayed(get_fit_prior)(fit, scoreset, **kwargs)
                               for fit in fits))
    point_values = kwargs.get('point_values',[1,2,3,4,5,6,7,8])
    prior = np.nanmedian(fit_priors)
    if np.isnan(prior) or prior == 0 or prior == 1:
        raise ValueError(f"Invalid prior estimate {prior} for scoreset {scoreset.scoreset_name}")
    observed_scores = scoreset.scores[scoreset._sample_assignments.any(1)]
    score_range = np.linspace(*np.percentile(observed_scores,[0,100]),10000) # type: ignore

    _log_fp = np.stack([density_utils.mixture_pdf(score_range, _fit['fit']['component_params'],_fit['fit']['weights'][0])
                        for _fit in fits])
    _log_fb = np.stack([density_utils.mixture_pdf(score_range, _fit['fit']['component_params'],_fit['fit']['weights'][1])
                       for _fit in fits])
    log_fp = np.full((len(fits),len(score_range)),np.nan)
    log_fb = np.full((len(fits),len(score_range)),np.nan)
    for fitIdx,(fit, fp,fb) in enumerate(zip(fits, _log_fp,_log_fb)):
        fit_xmin,fit_xmax = fit['fit']['xlims']
        mask = (score_range >= fit_xmin) & (score_range <= fit_xmax)
        log_fp[fitIdx,mask] = fp[mask]
        log_fb[fitIdx,mask] = fb[mask]
    
    
    log_lr_plus = log_fp - log_fb
    nan_counts = np.isnan(log_lr_plus).sum(0)
    range_subset = nan_counts < log_lr_plus.shape[0]

    point_ranges_pathogenic, point_ranges_benign = calculate_score_ranges(np.nanpercentile(log_lr_plus[:,range_subset],
                                                                                           5,axis=0),
                                                                            np.nanpercentile(log_lr_plus[:,range_subset],
                                                                                             95,axis=0),
                                                                            prior,
                                                                            score_range[range_subset],
                                                                            point_values,)
    point_ranges = {**point_ranges_pathogenic,**point_ranges_benign}
    return fit_priors, point_ranges,score_range[range_subset],log_fp[:,range_subset], log_fb[:,range_subset]
    


def get_fit_prior(fit, scoreset,**kwargs):
    pathogenic_idx = kwargs.get('pathogenic_idx',0)
    benign_idx = kwargs.get('benign_idx',1)
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']
    population = scoreset.scores[scoreset._sample_assignments[:,2]]
    pathogenic_density = density_utils.joint_densities(population,
                                                       params,
                                                       weights[pathogenic_idx]).sum(axis=0)
    benign_density = density_utils.joint_densities(population,
                                                       params,
                                                       weights[benign_idx]).sum(axis=0)
    assert len(pathogenic_density) == len(population)
    assert len(benign_density) == len(population)
    prior_estimate = 0.5
    converged = False
    tolerance = kwargs.get("tolerance", 1e-4)
    em_steps = 0
    max_em_steps = kwargs.get("max_em_steps",10000)
    while not converged:
        em_steps += 1

        with np.errstate(divide='ignore', invalid='ignore', over='ignore',under='ignore'):
            posteriors = 1 / (
                1
                + (1 - prior_estimate)
                / prior_estimate
                * benign_density # type: ignore
                / pathogenic_density
            )
        new_prior = np.nanmean(posteriors)
        prior_estimate = new_prior
        if prior_estimate < 0 or prior_estimate > 1:
            raise ValueError(f"Invalid prior estimate obtained, {prior_estimate}")
        if em_steps >= max_em_steps:
            break
    return prior_estimate
    
def plot_scoreset(scoreset:Scoreset, summary: Dict, scoreset_fits: List[Dict], score_range):
    fig, ax = plt.subplots(2,scoreset.n_samples, figsize=(5*scoreset.n_samples,10),sharex=True,sharey=False)
    for sample_num in range(scoreset.n_samples):
        sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]],stat='density',ax=ax[1,sample_num],alpha=.5,color='pink',)
        density = sample_density(score_range, scoreset_fits, sample_num)
        for compNum in range(density.shape[1]):

            compDensity = density[:,compNum,:]
            d = np.nanpercentile(compDensity,[5,50,95],axis=0)
            ax[1,sample_num].plot(score_range,d[1],color=f"C{compNum}",linestyle='--',label=f"Comp {compNum+1}")
            ax[1,sample_num].legend()
        d = np.nansum(density,axis=1)
        d_perc = np.percentile(d,[5,50,95],axis=0)
        ax[1,sample_num].plot(score_range,d_perc[1],color='black',alpha=.5)
        ax[1,sample_num].fill_between(score_range,d_perc[0],d_perc[2],color='gray',alpha=0.3)
        ax[1,sample_num].set_xlabel("Score")
        ax[0,sample_num].set_title(f"{scoreset.sample_names[sample_num]} (n={scoreset.sample_assignments[:,sample_num].sum():,d})")
    point_ranges = sorted([(int(k), v) for k,v in summary['point_ranges'].items()])
    point_values = [pr[0] for pr in point_ranges]
    for axi in ax[0]:
        for pointIdx,(pointVal, scoreRanges) in enumerate(point_ranges):
            for sr in scoreRanges:
                axi.plot([sr[0], sr[1]], [pointIdx,pointIdx], color='red' if pointVal > 0 else 'blue', linestyle='-', alpha=0.7)
        axi.set_ylim(-1,len(point_values))
        axi.set_ylabel("Points")

        axi.set_yticks(range(len(point_values)),labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0",point_values)))
    ax[0,2].set_title(f"{scoreset.population_type} (n={scoreset.sample_assignments[:,2].sum():,d})\nprior {summary['prior']:.3f}")
    return fig

def sample_density(x, fits, sampleNum):
    _density = np.stack([density_utils.joint_densities(x, _fit['fit']['component_params'],_fit['fit']['weights'][sampleNum])
                        for _fit in fits])
    density = np.full(_density.shape,np.nan)
    for fitIdx,fit in enumerate(fits):
        fit_xmin,fit_xmax = fit['fit']['xlims']
        mask = (x >= fit_xmin) & (x <= fit_xmax)
        density[fitIdx,:,mask] = _density[fitIdx,:,mask]
    return density

def add_thresholds(tauP, tauB, ax):
    for tp,tb in zip(tauP,tauB):
        ax.axhline(tp,color='red',linestyle='--',alpha=0.5)
        ax.axhline(tb,color='blue',linestyle='--',alpha=0.5)

def plot_summary(scoreset: Scoreset, fits: List[Dict], summary:Dict, score_range, log_fp, log_fb):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    log_lr_plus = log_fp - log_fb
    llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
    labels = ['5th percentile','Median','95th percentile']
    for i,c in enumerate(['red','black','blue']):
        ax.plot(score_range,llr_curves[i],color=c,label=labels[i])
    point_values = sorted(list(set([abs(int(k)) for k in summary['point_ranges'].keys()])))
    tauP,tauB = list(map(np.log, thresholds_from_prior(summary['prior'],point_values)) )
    priors = np.percentile(np.array(summary['priors']),[5,50,95])
    ax.set_title(f"Prior: {priors[1]:.3f} ({priors[0]:.3f}-{priors[2]:.3f})")
    add_thresholds(tauP, tauB, ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Log LR+")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig

def stable_alpha(beta, Log_LRs, eps=1e-12, min_lr=1e-12):
    """
    Compute alpha = averaged posterior probability given prior beta and positive LRs.
    Uses log-odds for numerical stability.

    Args:
      beta: current prior probability in (0,1) (float)
      Log_LRs: fixed iterable of positive likelihood ratios (floats)
      eps: small value to clamp beta away from 0/1
      min_lr: small positive value to clamp LRs away from 0

    Returns:
      alpha: posterior probability in (0,1)
    """
    # raise NotImplementedError("This function has not been tested yet.")
    # clamp beta into (eps, 1-eps)
    beta = min(max(beta, eps), 1.0 - eps)

    # safe logit: log(beta) - log(1-beta) using log1p for stability
    logit_beta = math.log(beta) - math.log1p(-beta)

    # # sum logs of LRs, clamping any non-positive LR to min_lr
    # s = 0.0
    # for llr in Log_LRs:
    #     if llr <= 0.0:
    #         lr = min_lr
    #     s += math.log(lr)
    s = np.sum(Log_LRs)

    # log-odds of posterior
    log_odds_post = logit_beta + Log_LRs

    # stable sigmoid: avoid overflow for large +/- values
    mask_gt = log_odds_post >= 0
    posts = np.zeros_like(Log_LRs)
    posts[mask_gt] = 1.0 / (1.0 + np.exp(-log_odds_post[mask_gt]))
    posts[~mask_gt] = np.exp(log_odds_post[~mask_gt]) / (1.0 + np.exp(log_odds_post[~mask_gt]))
    # if log_odds_post >= 0:
    #     z = math.exp(-log_odds_post)
    #     alpha = 1.0 / (1.0 + z)
    # else:
    #     z = math.exp(log_odds_post)
    #     alpha = z / (1.0 + z)
    alpha = np.mean(posts)
    return alpha

def basic_alpha(beta, benign_density, pathogenic_density):
    posteriors = 1 / (
            1
            + (1 - beta)
            / beta
            * benign_density # type: ignore
            / pathogenic_density
        )
    return np.nanmean(posteriors)

def test_stable():
    beta = 1-1e-6
    scores = np.concatenate((sps.norm.rvs(loc=-12,scale=2,size=1000000),
                             sps.norm.rvs(loc=12,scale=2,size=900)))

    Log_LRS = sps.norm.logpdf(scores,loc=-3,scale=2) - \
                sps.norm.logpdf(scores,loc=3,scale=2)
    alphaHatStable = stable_alpha(beta,Log_LRS)
    path_density = sps.norm.pdf(scores,loc=-3,scale=2)
    ben_density = sps.norm.pdf(scores,loc=3,scale=2)
    alphaHatBasic = basic_alpha(beta, ben_density, path_density)
    assert np.isclose(alphaHatBasic,alphaHatStable), f"Basic Estimate ({alphaHatBasic:.5f}) not close to Stable Estimate ({alphaHatStable:.5f})"
