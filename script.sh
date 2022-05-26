#!/bin/bash

#SBATCH --mail-user=
#SBATCH --mail-type=end,fail

#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --output=./slurms/test_with_preds_%A_%a.out
#SBATCH --array=1


#param_store=$HOME/args.txt

# Get first argument
#dataset=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

# Put your code below this line
#module load Python/3.9
#cd $HOME/graph_matching/graph-matching-gnn-reduction
#source venv/bin/activate

srun python main.py #--dataset $dataset

