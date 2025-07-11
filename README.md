# Assay Calibration Project

This repository contains resources for fitting models as part of the **Pillar Project** assay calibration process.

## Project Overview

This repository contains the code for fitting the models to pre-processed data in the format of the pillar project dataframe. Results can be summarized using the [`calibration_summary_utils repo`](https://github.com/Dzeiberg/calibration_summary_utils)

## Repository Structure

- `src/` - Source code for loading data and fitting calibration models
- `test/` - Includes example script for running calibration

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/Dzeiberg/assay_calibration.git
    ```
2. Create a conda environment with Python 3.10:
    ```bash
    conda create -n assay_calibration_env python=3.10
    conda activate assay_calibration_env
    ```
3. Install the package:
    ```bash
    pip install assay_calibration
    ```

## Usage

### Example using Pillar Project Dataframe
Example usage
```python
from assay_calibration.data_utils.dataset import Scoreset
from assay_calibration.fit_utils.fit import Fit
import pandas as pd
import json

dataframe = pd.read_csv("test/example_data.csv")
scoreset = Scoreset(dataframe)
fit = Fit(scoreset)
fit.run(core_limit=1, num_fits=1, component_range=[2, 3])
result = fit.to_dict()
print(json.dumps(result, indent=4))
```
### Example using other data

**Input File Format**
```csv
scores,sample_assignments
0.1,sample0
0.2,sample1
0.15,sample2
0.07,sample0
0.17,sample2
```

```python
from assay_calibration.data_utils.dataset import BasicScoreset
from assay_calibration.fit_utils.fit import Fit
import pandas as pd
import json

scoreset = BasicScoreset.from_csv("test/example_table.csv")
fit = Fit(scoreset)
fit.run(core_limit=1, num_fits=1, component_range=[2, 3])
result = fit.to_dict()
print(json.dumps(result, indent=4))
```
### Hyper-parameters

The `num_fits` hyper-parameter encodes how many times model fitting should be repeated for each of the component number candidates. The final fit is that with the maximum likelihood estimate on a held-out validation set. Each model fit starts with a random initialization, so increasing `num_fits` often results in better models. Experiments were run with `num_fits=100`.

The `component_range` hyper-parameter encodes the possible number of skew-normal components used to model the data. Data exploration, visual inspection, and domain knowledge can help in choosing these values. Consecutive components are constrained such that the likelihood ratio is monotonic, preventing a wide component from overtaking the other component at extreme values.

Uncertainty in the distribution fits can be obtained by fitting models to many bootstrapped scoresets, i.e., set of points sampled with replacement from the full scoreset.
## Contributing

Contributions are welcome! Please submit issues or pull requests to help improve the project.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please contact [d.zeiberg@northeastern.edu](mailto:d.zeiberg@northeastern.edu).