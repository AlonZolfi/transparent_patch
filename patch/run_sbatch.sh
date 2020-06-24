#!/bin/bash

### sbatch config parameters must start with #SBATCH, to ignore just add another # - like ##SBATCH

#SBATCH --partition short		### specify partition name where to run a job
#SBATCH --qos shabtaia		### priority

##SBATCH --time 1-11:00:00			### limit the time of job running, partition limit can override this

#SBATCH --job-name patch-train			### name of the job
#SBATCH --output /home/zolfi/jobs/job-%J.log			### output log for running job - %J for job number
##SBATCH --mail-user=zolfi@post.bgu.ac.il	### users email for sending job status
##SBATCH --mail-type=ALL		### conditions when to send the email

#SBATCH --gres=gpu:1				### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU

### Start you code below ####

module load anaconda ### load anaconda module
source activate adversarial-yolo ### activating environment, environment must be configured before running the job (conda)
python train.py