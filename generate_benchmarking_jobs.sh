#!/bin/bash
#SBATCH --job-name=generate_jobs
#SBATCH --output=/projects/talisman/d.zeiberg/logs/slurm-%j.out
#SBATCH --error=/projects/talisman/d.zeiberg/logs/slurm-%j.err
#SBATCH --time=03:59:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

module load anaconda3/2024.06
conda activate assay_calibration

python benchmarking/generate_benchmark_jobs.py \
/projects/talisman/dzeiberg/assay_calibration/benchmarking/dataframe_expanded.csv.tar.gz \
/projects/talisman/dzeiberg/assay_calibration/benchmarking/fits/ \
/projects/talisman/dzeiberg/assay_calibration/benchmarking/jobs