#!/bin/bash
#SBATCH --job-name=calibration_summary
#SBATCH --output=/projects/talisman/d.zeiberg/logs/slurm-%j.out
#SBATCH --error=/projects/talisman/d.zeiberg/logs/slurm-%j.err
#SBATCH --time=23:59:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-49
module load anaconda3/2024.06
conda activate assay_calibration

# --- CONFIGURATION ---
FILES_DIR="/projects/talisman/dzeiberg/tsc2/tsc2_jobs/*.pkl"
"

# Get the list of all files (change pattern as needed)
FILELIST=$(find "$FILES_DIR" -type f | sort)
TOTAL_FILES=$(echo "$FILELIST" | wc -l)
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT}
JOB_ID=${SLURM_ARRAY_TASK_ID}

# Calculate chunk size (number of files per job)
CHUNK_SIZE=$(( (TOTAL_FILES + NUM_JOBS - 1) / NUM_JOBS ))

# Compute start and end indices for this job
START=$(( JOB_ID * CHUNK_SIZE + 1 ))
END=$(( (JOB_ID + 1) * CHUNK_SIZE ))

# Select the subset of files for this job
FILES=$(echo "$FILELIST" | sed -n "${START},${END}p")

# --- PROCESS FILES ---
i=0
while read -r FILE; do
    [ -z "$FILE" ] && continue
    BASENAME=$(basename "$FILE")
    OUTFILE="$OUTPUT_DIR/${BASENAME%.txt}_out.txt"  # adjust as needed

    # Run your command
    # $COMMAND "$FILE" > "$OUTFILE"
    python execute_job.py "$FILE"

    ((i++))
done <<< "$FILES"

echo "Job $JOB_ID processed $i files ($START-$END)."