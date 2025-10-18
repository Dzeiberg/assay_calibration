import pandas as pd
from dataset import Scoreset
from argparse import Namespace
pd.set_option("display.max_columns",None)

df = pd.read_csv("/data/projects/igvf/assay_calibration/dataframe_expanded.csv.tar.gz")

scoreset_df = df[df.Dataset == "BRCA1_Findlay_2018"]

scoreset = Scoreset(scoreset_df,clinvar_release='2018',min_clinvar_star=0)