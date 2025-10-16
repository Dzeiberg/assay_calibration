from test_fit_figures import main as make_figures
from pathlib import Path
from tqdm.auto import tqdm
from argparse import Namespace
import pandas as pd

figures_rt = Path("/data/dzeiberg/assay_calibration_population_selection/test_figures_2/")
figures_rt.mkdir(exist_ok=True,parents=True)
fits_rt = Path("/data/dzeiberg/assay_calibration_population_selection/test_fits/")
pillar_df_filepath = "/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/final_pillar_data_with_clinvar_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_091125.csv.tar.gz"
df = pd.read_csv(pillar_df_filepath)

scoreset_names = [f.stem for f in fits_rt.iterdir() if f.is_dir()]
for scoreset in tqdm(scoreset_names, desc="generating scoreset figures"):
    # try:
    make_figures(Namespace(figure_savedir=figures_rt / scoreset,
                    fits_directory=fits_rt/scoreset,
                    df=df,
                    scoreset_name=scoreset))
    # except Exception as e:
    #     print(e)
    #     continue