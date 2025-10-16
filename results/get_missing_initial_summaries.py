from get_pillar_project_summaries import get_summaries

if __name__ == "__main__":
    get_summaries("/data/projects/igvf/assay_calibration/initial_datasets_results_1000bootstraps_100fits.json.gz",
                  "/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz",
                  "/data/projects/igvf/assay_calibration/results/initial_datasets_results_1000bootstraps_100fits/summaries_v6/",
                  scoreset_names=['CTCF_unpublished',
                                  "GCK_Gersing_2023_complementation",
                                  "HMBS_van_Loggerenberg_2023_ubquitous",
                                  "KCNH2_Kozek_Glazer_2020",
                                  "SCN5A_Glazer_2020",
                                  "TP53_Boettcher_2019",
                                  "TP53_Kato_2003_AIP1nWT",])