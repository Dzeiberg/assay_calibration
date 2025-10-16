import pandas as pd
import argparse
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parents[1]))
from src.assay_calibration.fit_utils.fit import (Fit,calculate_score_ranges,assign_points)
from src.assay_calibration.data_utils.dataset import Scoreset
from src.assay_calibration.fit_utils.two_sample import density_utils

def main(args):
    if "df" in args.__dict__:
        df = args.df
    else:
        df = pd.read_csv(args.pillar_df_filepath)
    scoresets = make_scoresets(df,args)
    scoreset_fits = process_fits(scoresets, args)
    scoreset_priors = get_priors(scoreset_fits,scoresets,args)
    point_score_ranges, scoreset_log_lrPlus = get_thresholds(scoresets, scoreset_fits,scoreset_priors)
    point_distributions = get_sample_point_distributions(scoresets, point_score_ranges)
    plot_log_lrs(scoreset_log_lrPlus, args)
    make_clinvar_point_distribution_figures(point_distributions,scoreset_priors,args)
    make_fits_figure(scoresets, scoreset_fits,args.scoreset_name,scoreset_priors,args)
    make_point_distribution_figures(point_distributions,args)
    

def get_fit_prior(fit,scoreset):
    scores = scoreset.scores
    sample_assignment = scoreset.sample_assignments
    population_sample = scores[sample_assignment[:,2]]
    try:
        prior = Fit.from_dict(scoreset,fit['fit']).get_prior_estimate(population_sample,
                                                         tolerance=1e-5)
    except ValueError as e:
        print(e)
        prior = np.nan
    return prior

def get_priors(fits,scoresets,args):
    scoreset_priors = {}
    for population_type, population_fits in fits.items():
        scoreset_priors[population_type] = np.array([get_fit_prior(fit,
                                                                   scoresets[population_type]) for fit in population_fits])
    return scoreset_priors

def make_scoresets(df, args):
    scoresets = {population_type : Scoreset(df[df.Dataset == args.scoreset_name],
                                        population_type=population_type) \
                                            for population_type in ['all_variants',
                                                                        'all_nsSNV',
                                                                        'all_missense_nsSNV',
                                                                        'gnomAD',
                                                                        'gnomAD_nsSNV',
                                                                        'gnomAD_missense_nsSNV']}
    return scoresets

def process_fits(scoresets, args):
    fits_dir = Path(args.fits_directory)
    def process_fit(population_type,fit_filepath):
        with open(fit_filepath) as f:
            fit = json.load(f)
        fit['fit']['weights'] = np.array(fit['fit']['weights'])
        return ((population_type,fit['bootstrap_seed']),fit)


    processed_fits = Parallel(n_jobs=-1, verbose=100)(delayed(process_fit)(population_type, fit_file) \
                                                    for population_type in scoresets.keys() \
                                                        for fit_file in tqdm(list((fits_dir / population_type).glob("*.json"))))
    sorted_processed_fits = sorted(processed_fits, key=lambda tup: -tup[1]['val_ll']) # type: ignore
    scoreset_fits = {population_type: dict() for population_type in scoresets.keys()}
    for (population_type, bootstrap_index), fit in sorted_processed_fits: # type: ignore
        if bootstrap_index not in scoreset_fits[population_type]:
            scoreset_fits[population_type][bootstrap_index] = fit
    scoreset_fits = {k : list(d.values()) for k,d in scoreset_fits.items()}
    return scoreset_fits

def make_fits_figure(scoresets,scoreset_fits,scoreset_name,population_priors,args):
    figdims = np.array([scoresets['gnomAD'].sample_assignments.shape[1],
                    len(scoresets)])
    fig,ax = plt.subplots(*figdims, figsize=np.array((8,3)) * figdims,sharex=True,sharey='row')

    for _,((population_type,scoreset),scoresetAx) in enumerate(zip(scoresets.items(),
                                                                            ax.T)): # type: ignore
        scoreset_fit = scoreset_fits[population_type]
        for i,(sample_scores, sample_name) in enumerate(scoreset.samples):
            score_range = np.linspace(sample_scores.min(),sample_scores.max(),1000)
            sns.histplot(sample_scores, ax=scoresetAx[i],stat='density',label=f"n={len(sample_scores):,d}")
            scoresetAx[i].set_title(sample_name.replace("population",scoreset.population_type))
            pdf = np.stack([density_utils.joint_densities(score_range,
                                                fit['fit']['component_params'],
                                                fit['fit']['weights'][i]).sum(0) for fit in scoreset_fit])
            
            scoresetAx[i].plot(score_range, np.median(pdf,axis=0),color='C1')
            scoresetAx[i].legend()
            scoresetAx[i].fill_between(score_range, *np.percentile(pdf,[2.5,97.5],axis=0),color='C1',alpha=.5)
        population_idx = [i for i,name in enumerate(scoreset.sample_names) if name in {'population',population_type}][0]
        scoresetAx[population_idx].set_title(f"{scoreset.population_type}\nprior: {np.median(population_priors[population_type]):.3f}")
    save_dir = Path(args.figure_savedir)
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / f"{scoreset_name}_distributions.png",dpi=300)
    plt.close(fig)

def plot_log_lrs(population_log_lrPlus,args):
    fig,ax = plt.subplots(1,1)
    for population_name, popvals in population_log_lrPlus.items():
        # low,med,high = np.percentiles()
        plt.plot(popvals['scores'], np.median(popvals['log_lr'],axis=0),label=population_name)
    ax.set_xlabel("Score")
    ax.set_ylabel("log LR+")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    save_fp = Path(args.figure_savedir) / "lr_curves.png"
    save_fp.parent.mkdir(exist_ok=True,parents=True)
    fig.savefig(save_fp, dpi=300,bbox_inches='tight')
    plt.close(fig)

def get_thresholds(scoresets, scoreset_fits,population_priors):
    population_point_score_ranges = {}
    population_log_lrPlus = {}
    for population_type,scoreset in scoresets.items():
        score_range = np.linspace(*np.percentile(scoreset.scores,[0,100]),10000) # type: ignore
        log_fp = np.stack([density_utils.mixture_pdf(score_range, _fit['fit']['component_params'],_fit['fit']['weights'][0])
                        for _fit in scoreset_fits[population_type]])
        log_fb = np.stack([density_utils.mixture_pdf(score_range, _fit['fit']['component_params'],_fit['fit']['weights'][1])
                        for _fit in scoreset_fits[population_type]])
        log_lr_plus = log_fp - log_fb
        score_ranges_pathogenic, score_ranges_benign = calculate_score_ranges(np.percentile(log_lr_plus,5,axis=0),
                                                                            np.percentile(log_lr_plus,95,axis=0),
                                                                            np.median(population_priors[population_type]),
                                                                            score_range,
                                                                            [1,2,3,4,8],)
        population_point_score_ranges[population_type] = {**score_ranges_pathogenic,
                                                        **score_ranges_benign}
        population_log_lrPlus[population_type] = {"scores":score_range,"log_lr":log_lr_plus}
    return population_point_score_ranges, population_log_lrPlus

def get_point_distribution(point_assignment, sample_assignment,sample_names):
    sample_point_distrs = {}
    for sample_idx, sample_name in enumerate(sample_names):
        sample_mask = np.where(sample_assignment[:,sample_idx])[0]
        sample_points = point_assignment[sample_mask]
        sample_point_distrs[sample_name] = dict(list(zip(*np.unique(sample_points,return_counts=True))))
    return sample_point_distrs

def get_sample_point_distributions(scoresets,population_point_score_ranges):
    point_values = [-8,-4,-3,-2,-1,0,1,2,3,4,8]
    sample_point_dfs = {}
    for sample_name in scoresets.keys():
        point_assignments = assign_points(scoresets[sample_name].scores,
                                        population_point_score_ranges[sample_name])
        _point_distr = get_point_distribution(point_assignments,
                                            scoresets[sample_name].sample_assignments,
                                            scoresets[sample_name].sample_names)
        _point_distr[sample_name] = _point_distr.pop("population")
        _point_df = pd.DataFrame.from_dict(_point_distr,orient='index')
        for p in point_values:
            if p not in _point_df.columns:
                _point_df[p] = 0
        _point_df = _point_df.loc[:,point_values]
        _point_df = _point_df.div(_point_df.sum(axis=1), axis=0).fillna(0)
        sample_point_dfs[sample_name] = _point_df
    return sample_point_dfs

def make_point_distribution_figures(sample_point_dfs,args):
    save_dir = Path(args.figure_savedir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for sample_name, _point_df in sample_point_dfs.items():
        fig, ax = plt.subplots(1,1,figsize=(12,5))
        sns.heatmap(data=_point_df, cmap="crest", vmin=0, vmax=1, annot=True, fmt=".2%", ax=ax)
        fig.savefig(save_dir / f"point_distribution_{sample_name}.png",
                    dpi=300,bbox_inches='tight')
        plt.close(fig)

def make_clinvar_point_distribution_figures(sample_point_dfs, population_priors,args):
    colors = ['#67001f',
              '#b2182b',
              '#d6604d',
              '#f4a582',
              '#fddbc7',
              '#f7f7f7',
              '#d1e5f0',
              '#92c5de',
              '#4393c3',
              '#2166ac',
              '#053061']
    color_map = dict(zip([8,4,3,2,1,0,-1,-2,-3,-4,-8],colors))
    
    scoreset_name = args.scoreset_name
    save_dir = Path(args.figure_savedir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plp_df = pd.DataFrame([df.loc['Pathogenic/Likely Pathogenic'] for df in sample_point_dfs.values()],
                        index=[sampleName for sampleName in sample_point_dfs.keys()])

    blb_df = pd.DataFrame([df.loc['Benign/Likely Benign'] for df in sample_point_dfs.values()],
                        index=[sampleName for sampleName in sample_point_dfs.keys()])
    blb_df = blb_df.loc[:,[c for c in blb_df.columns if (int(c) < 0) or (blb_df[c].sum() > 0)]]
    plp_df = plp_df.loc[:,[c for c in plp_df.columns if (int(c) > 0) or (plp_df[c].sum() > 0)]]
    # Pathogenic/Likely pathogenic figure
    ax = plp_df.iloc[:,::-1].plot(kind='barh', stacked=True, color=color_map)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yticklabels([f"{sample_name.replace('_',' ')}\nprior: {np.median(population_priors[sample_name]):.3f}"\
                        for sample_name in plp_df.index])
    ax.set_xlabel("Fraction of Variants")
    ax.set_title("Pathogenic/Likely pathogenic point distribution")
    ax.set_ylabel("Population")
    fig = plt.gcf()
    fig.savefig(save_dir / f"{scoreset_name}_pathogenic_point_distributions.png",
                dpi=300,bbox_inches='tight')
    plt.close(fig)
    # Benign Likely benign figure
    ax = blb_df.plot(kind='barh', stacked=True, color=color_map)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_yticklabels([f"{sample_name.replace('_',' ')}\nprior: {np.median(population_priors[sample_name]):.3f}"\
                        for sample_name in blb_df.index])
    ax.set_xlabel("Fraction of Variants")
    ax.set_title("Benign/Likely benign point distribution")
    ax.set_ylabel("Population")
    fig = plt.gcf()
    fig.savefig(save_dir / f"{scoreset_name}_benign_point_distributions.png",
                dpi=300,bbox_inches='tight')
    plt.close(fig)



if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        args = argparse.Namespace(**dict(scoreset_name="BRCA1_Findlay_2018",
         figure_savedir="/data/dzeiberg/assay_calibration_population_selection/test_figures/BRCA1_Findlay_2018",
         fits_directory="/data/dzeiberg/assay_calibration_population_selection/test_fits/BRCA1_Findlay_2018/",
         pillar_df_filepath="/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz"))
    else:
        parser = argparse.ArgumentParser(description="Process some assay calibration data.")
        parser.add_argument(
            "scoreset_name",
            type=str,
            help="Name of the scoreset to process."
        )
        parser.add_argument(
            "figure_savedir",
            type=str,
            help="Directory to save the generated figures."
        )
        parser.add_argument(
            "fits_directory",
            type=str,
            help="Directory containing model fits."
        )
        parser.add_argument(
            "--pillar_df_filepath",
            type=str,
            default="/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz",
            help="Filepath to the pillar dataframe."
        )

        args = parser.parse_args()
    main(args)
    