import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.assay_calibration.data_utils.dataset import PillarProjectDataframe, Scoreset
from src.assay_calibration.fit_utils.fit import Fit
import json


def test_fit():
    df = PillarProjectDataframe(Path(__file__).parent / "example_data.csv")
    ds = Scoreset(df.dataframe[df.dataframe.Dataset == "BRCA1_Adamovich_2022_HDR"])
    print(ds)
    fit = Fit(ds)
    fit.run(core_limit=1, num_fits=1, component_range=[2, 3])
    result = fit.to_dict()
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    test_fit()
    print("Test completed successfully.")
