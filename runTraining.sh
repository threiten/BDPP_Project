#!/bin/bash

#SBATCH --job-name=train_LSTMMultiClass
#SBATCH --account=gpu_gres
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=12288
#SBATCH --gres=gpu:1

source /work/threiten/anaconda/bin/activate /work/threiten/conda-envs/bdppPCT

mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}
mkdir -p ${TMPDIR}/out

cp -r /work/threiten/BDPP_Data/Corp_Bundestag_V2.zarr $TMPDIR/

echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES

export CUBLAS_WORKSPACE_CONFIG=:4096:2
# export CUDA_LAUNCH_BLOCKING=1

echo SLURM_ARRAY_TASK_ID : ${SLURM_ARRAY_TASK_ID}

python3 /t3home/threiten/BDPP_Project/runTraining.py -c $CUDA_VISIBLE_DEVICES -i ${TMPDIR}/Corp_Bundestag_V2.zarr -s ${TMPDIR}/out/LSTMMultiClass_trained.pt -t ${TMPDIR} --config ${SLURM_ARRAY_TASK_ID}

cp -r ${TMPDIR}/out /work/threiten/BDPP_Data/TrainedModels/outdir_${SLURM_JOB_ID}

rm -rf /scratch/${USER}/${SLURM_JOB_ID} 
