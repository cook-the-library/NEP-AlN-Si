#!/bin/bash
#SBATCH -J r1_generate
#SBATCH -o round1_out/logs/z-generate-%j.out
#SBATCH -t 0:10:0
#SBATCH -N 1 --ntasks-per-node=1
#SBATCH -A mat260051

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dokwankook@gmail.com

# ---------------------------------------------------------------------------
# Step 1 of round 1: generate the structures and the VASP job list.
#
# Submit from the PROJECT ROOT (the folder holding round1_code/ and
# aaa-potential/). The log folder has to exist first, because sbatch opens
# the -o file before this script runs:
#
#     mkdir -p round1_out/logs
#     sbatch round1_code/sbatch_round1.sbatch
#
# Writes:
#     round1_out/vasp/<bucket>_<index>/     one VASP folder per structure
#     round1_out/round1_1_vasp_job_paths.txt
# ---------------------------------------------------------------------------

#Start from the directory that I submitted this job from
cd $SLURM_SUBMIT_DIR
[ -d round1_code ] || { echo "submit from the project root: no round1_code/ in $PWD"; exit 1; }
mkdir -p round1_out

#Load the necessary modules
#module use /storage/home/hcoda1/1/dkook3/mylammps/src
module load conda
conda-env-mod create -n mypackages -y
module use $HOME/privatemodules
module load conda-env/mypackages
pip install ase
pip install pymatgen
pip install numpy

python round1_code/generate_round1.py --outdir round1_out/vasp

find ./round1_out/vasp/*/ -name "POSCAR" -exec dirname {} \; > round1_out/round1_1_vasp_job_paths.txt
