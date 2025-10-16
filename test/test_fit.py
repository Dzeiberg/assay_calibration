import sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.assay_calibration.data_utils.dataset import (
    PillarProjectDataframe,
    Scoreset,
    BasicScoreset,
)
from src.assay_calibration.fit_utils.fit import Fit
import json
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def test_fit(cpu_limit=-1,N_restarts=100,num_bootstrap_iters=10):
    scoresets_dir = Path('/data/dzeiberg/pillar_project/pillar_project_data/dataset_09192025/scoresets')
    jobs = []
    for scoreset_file in tqdm(scoresets_dir.glob("*.json"),desc="Generating scoreset jobs"):
        scoreset = Scoreset.from_json(scoreset_file)
        fit = Fit(scoreset)
        save_dir = Path(__file__).parent / "test_fits" / scoreset.scoreset_name
        save_dir.mkdir(exist_ok=True,parents=True)
        scoreset_jobs = sum([fit.generate_fit_jobs([2,3],
                                        save_dir,
                                        bootstrap_seed=fit_num,
                                        num_fits=N_restarts) \
                        for fit_num in range(num_bootstrap_iters)],[])
        jobs += scoreset_jobs
    print(f"Running {len(jobs):,d} jobs using {cpu_limit} cores")
    if cpu_limit == 1:
        for job in jobs:
            Fit.execute_fit_job(job)
    else:
        Parallel(n_jobs=cpu_limit,verbose=100)(delayed(Fit.execute_fit_job)(job) for job in jobs)

def test_basic_scoreset():
    abnormal = stats.norm(loc=-5, scale=3)
    normal = stats.norm(loc=0, scale=3)
    scores = np.concatenate([abnormal.rvs(100), normal.rvs(150), abnormal.rvs(50)])
    sample_assignments = np.zeros_like(scores)
    sample_assignments[:100] = 1
    sample_assignments[100:200] = 2
    sample_assignments[200:] = 1
    scoreset = BasicScoreset(scores, sample_assignments)
    fit = Fit(scoreset)
    fit.run(core_limit=5, num_fits=5, component_range=[2, 3], verbose=True, verbose_level=20)
    result = fit.to_dict()
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    test_fit()
    print("Test completed successfully.")
