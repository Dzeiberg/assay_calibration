
import pandas as pd
from tqdm.auto import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.assay_calibration.fit_utils.fit import (Fit,calculate_score_ranges, thresholds_from_prior)
from src.assay_calibration.data_utils.dataset import Scoreset
from src.assay_calibration.fit_utils.two_sample import density_utils

import numpy as np
import logging
from typing import List
import json
import pickle
from joblib import Parallel, delayed
pd.set_option("display.max_columns",None)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz")

def write_scoreset_jobs(scoreset_name,jobs_save_rt,fits_save_rt,
                        N_restarts,num_bootstrap_iters):
    # # Generate jobs 
    scoreset_df = df[df.Dataset == scoreset_name]
    if scoreset_df.auth_reported_score.isna().all():
        print(f"all variants in {scoreset_name} have NaN author_reported_score")
        return
    def gen_scoreset(population_type):
        return Scoreset(scoreset_df,population_type=population_type)
    scoreset = gen_scoreset('gnomAD')
    fit = Fit(scoreset) # type: ignore
    # save_dir = fits_save_rt / scoreset.scoreset_name
    # if save_dir.exists() and any(save_dir.iterdir()):
    #     for file in save_dir.iterdir():
    #         file.unlink()
    fits_save_rt.mkdir(exist_ok=True,parents=True)
    # Generate jobs in parallel
    scoreset_jobs = Parallel(n_jobs=-1, verbose=True)(
        delayed(fit.generate_fit_jobs)([2, 3],
                                       fits_save_rt,
                                       bootstrap_seed=fit_num,
                                       num_fits=N_restarts,
                                       check_monotonic=check_monotonic)
        for fit_num in range(num_bootstrap_iters) for check_monotonic in [True,False]
    )
    # Flatten the list of lists
    scoreset_jobs = [job for sublist in scoreset_jobs for job in sublist] # type: ignore

    # Write jobs to file
    # job_savedir = jobs_save_rt / scoreset_name
    print(f"writing {len(scoreset_jobs)} jobs to {jobs_save_rt}")
    jobs_save_rt.mkdir(parents=True,exist_ok=True)
    Parallel(n_jobs=-1, verbose=True)(
        delayed(lambda jobNum, job: pickle.dump(job, open(jobs_save_rt / f"{scoreset_name}_job_{jobNum}.pkl", "wb")))(jobNum, job)
        for jobNum, job in enumerate(scoreset_jobs)
    )

scoreset_names = ['BARD1_unpublished',
                    'CTCF_unpublished',
                    'G6PD_unpublished',
                    'MSH2_Jia_2021',
                    'PALB2_unpublished',
                    'SFPQ_unpublished',
                    'TSC2_rapgap_unpublished',
                    'TSC2_tuberin_unpublished',
                    'OTC_Lo_2023',
                    'XRCC2_unpublished',
                    'RAD51D_unpublished',
                    'F9_Popp_2025_carboxy_F9_specific',
                    'F9_Popp_2025_carboxy_gla_motif',
                    'F9_Popp_2025_heavy_chain',
                    'F9_Popp_2025_light_chain',
                    'F9_Popp_2025_strep_2',
                    'TP53_Fortuno_2021_Kato_meta',
                    'TP53_Giacomelli_2018_combined_score',
                    'TP53_Giacomelli_2018_p53WT_Nutlin3',
                    'TP53_Giacomelli_2018_p53null_Nutlin3',
                    'TP53_Giacomelli_2018_p53null_etoposide',
                    'ASPA_Grønbæk-Thygesen_2024_abundance',
                    'ASPA_Grønbæk-Thygesen_2024_toxicity',
                    'CALM1_CALM2_CALM3_Weile_2017',
                    'CARD11_Meitlis_2020_DMSO_no_introns',
                    'CARD11_Meitlis_2020_Ibrutinib_no_introns',
                    'CBS_Sun_2020_high_B6',
                    'CBS_Sun_2020_low_B6',
                    'CHK2_Gebbia_2024',
                    'CRX_Shepherdson_2024',
                    'FKRP_Ma_2024',
                    'GCK_Gersing_2023_complementation',
                    'GCK_Gersing_2024_abundance',
                    'HMBS_van_Loggerenberg_2023_combined',
                    'HMBS_van_Loggerenberg_2023_erythroid',
                    'HMBS_van_Loggerenberg_2023_ubquitous',
                    'JAG1_Gilbert_2024',
                    'KCNE1_Muhammad_2024_absence_of_WT',
                    'KCNE1_Muhammad_2024_potassium_flux',
                    'KCNE1_Muhammad_2024_presence_of_WT',
                    'KCNH2_Jiang_2022',
                    'KCNH2_Kozek_Glazer_2020',
                    'KCNH2_O_Neill_2024_surface_expression',
                    'KCNQ4_Zheng_2022_current_homozygous',
                    'KCNQ4_Zheng_2022_v12_homozygous',
                    'LARGE1_Ma_2024',
                    'NDUFAF6_Sung_2024',
                    'PAX6_McDonnell_2024_BLX_geneticin',
                    'PAX6_McDonnell_2024_BLX_no_geneticin',
                    'PAX6_McDonnell_2024_LE9_geneticin',
                    'PAX6_McDonnell_2024_LE9_no_geneticin',
                    'RHO_Wan_2019',
                    'SCN5A_Glazer_2020',
                    'SCN5A_Ma_2024_current_density',
                    'SGCB_Li_2023',
                    'TARDBP_Bolognesi_Faure_2019',
                    'TP53_Boettcher_2019',
                    'TP53_Fayer_2021_meta',
                    'TP53_Kato_2003_AIP1nWT',
                    'TP53_Kato_2003_BAXnWT',
                    'TP53_Kato_2003_GADD45nWT',
                    'TP53_Kato_2003_MDM2nWT',
                    'TP53_Kato_2003_NOXAnWT',
                    'TP53_Kato_2003_P53R2nWT',
                    'TP53_Kato_2003_WAF1nWT',
                    'TP53_Kato_2003_h1433snWT',
                    'TPK1_Weile_2017',]

exclude_scoresets = set()
fits_save_rt = Path("/data/projects/igvf/assay_calibration/experiments/fits_14OCT2025")
jobs_save_rt = Path("/data/projects/igvf/assay_calibration/experiments/jobs_14OCT2025")
with tqdm(scoreset_names) as pbar:
    for scoreset_name in pbar:
        pbar.set_description(scoreset_name)
        write_scoreset_jobs(scoreset_name,jobs_save_rt,fits_save_rt,N_restarts=100,num_bootstrap_iters=1000)
        pbar.update(1)