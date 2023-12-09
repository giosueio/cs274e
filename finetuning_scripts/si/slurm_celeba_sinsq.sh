#!/bin/bash
#SBATCH --partition=datalab.p            # IMPORTANT -- Submit to our partition (datalab.p isn't the default so if you omit this you'll submit to a different ICS SLURM partition)
#SBATCH --mem=20gb                       # IMPORTANT -- Allocated RAM -- jobs that try to exceed this will be killed
#SBATCH --gpus=1                         # IMPORTANT -- Number of GPUs to be allocated (it will set CUDA_VISIBLE_DEVICES, don't override this) -- the GPUs will show up in your PyTorch code as starting with ID 0 always (then 1, 2, ...)
#SBATCH --job-name=stochint              # Job name (will show up under this in the queue)
#SBATCH --ntasks=1                       # Run a single task (generally don't change this, used  by MPI and OpenMP for parallel jobs)
#SBATCH --cpus-per-task=1                # Run on a single CPU thread (change this if you want multiple threads for multiprocessing) -- I don't believe this is currently enforced, but please only use as many as you ask for
#SBATCH --time=24:00:00                  # Time limit (hrs:min:sec) -- job will be killed if it exceeds this
#SBATCH --output=%x_%j.log  # Name of file that standard output and error will be stored (%j is the Job ID, %x is the Job Name)

# --- Print Pre-Job Info ---
pwd; hostname; date
echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES # You can print this to see which GPU(s) your job was allocated and look them up with nvidia-smi

# --- Run Pre-Job Setup ---

# You can load modules, setup the environment, etc.

# --- Run Job ---
echo -e "\n===== Job Start =====\n";

module load python3
python3 celeba.py

echo -e "\n===== Job End =====\n"

# By default python's 'print' will buffer its output so there will be a delay in your SLURM output log until the buffer fill or the job completes.
# You can use 'python -u main.py' to run unbuffered and see the outupt in the log almost immediately.

# If you're using conda (which I recommend), 'conda activate <env-name>' likely won't work.
# You can directly use the python binary for the correct environment (e.g. /home/<username>/miniconda3/envs/<env-name>/bin/python)
# OR if you submit the job with your conda env already active it should use that env for the job

# You can run jobs inside sbatch with srun (e.g. 'srun python main.py experiment.yml').
# This is most important for parallel jobs (e.g. using MPI and OpenMP), but can offer more granular reporting if desired.

# --- Post Job Info ---
date