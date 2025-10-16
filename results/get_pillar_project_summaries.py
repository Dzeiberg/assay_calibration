import logging
from summarize_scoreset import summarize_scoreset
from pathlib import Path
import gzip
import json
import pandas as pd
import sys
from tqdm.auto import tqdm
logging.getLogger('matplotlib').setLevel(logging.ERROR)
sys.path.append(str(Path(__file__).parents[1]))
from src.assay_calibration.data_utils.dataset import Scoreset  # noqa: E402

def get_summaries(fits_filepath: str|Path,
                  pillar_df_filepath: str|Path,
                  save_directory: str|Path,
                  **kwargs)->None:
    fits_filepath = Path(fits_filepath)
    if not fits_filepath.exists():
        raise ValueError(f"fits filepath {fits_filepath} does not exist")
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True,parents=True)
    pillar_df_filepath = Path(pillar_df_filepath)
    if not pillar_df_filepath.exists():
        raise ValueError(f"pillar_df_filepath {pillar_df_filepath} does not exist")
    with gzip.open(fits_filepath) as f:
        fits = json.load(f)
    
    df = pd.read_csv(pillar_df_filepath)
    scoreset_names = list(set(df.Dataset))
    scoreset_names = list(kwargs.get("scoreset_names",scoreset_names))
    fits = {k : fits[k] for k in scoreset_names if k in fits}
    with tqdm(list(fits.items()), desc="Summarizing scoresets") as pbar:
        for scoreset_name, scoreset_fits in pbar:
            pbar.update(1)
            pbar.set_postfix(dict(scoreset=scoreset_name))
            try:
                scoreset = Scoreset(df[df.Dataset == scoreset_name],
                                    population_type = 'gnomAD')
            except ValueError:
                print(f"Invalid scoreset name {scoreset_name}")
                continue
            sample_counts = scoreset._sample_assignments.sum(axis=0)
            if (sample_counts[:3] == 0).any():
                print(f"insufficient samples: {scoreset.sample_names}")
                print(f"sample counts: {sample_counts}")
                continue
            for component_key in ['2c','3c']:
                fits = [fit_iter[component_key] for fit_iter in scoreset_fits.values()]
                save_filepath = save_directory / scoreset_name / f"{component_key}.json"
                try:
                    summarize_scoreset(fits,scoreset,save_filepath)
                except ValueError as e:
                    print(e)
                    continue

if __name__ == "__main__":
    get_summaries("/data/projects/igvf/assay_calibration/clinvar_circ_datasets_results_1000bootstraps_100fits.json.gz",
                  "/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz",
                  "/data/projects/igvf/assay_calibration/results/clinvar_circ_datasets_results_1000bootstraps_100fits/summaries_v6/",)
    get_summaries("/data/projects/igvf/assay_calibration/initial_datasets_results_1000bootstraps_100fits.json.gz",
                  "/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz",
                  "/data/projects/igvf/assay_calibration/results/initial_datasets_results_1000bootstraps_100fits/summaries_v6/",)
