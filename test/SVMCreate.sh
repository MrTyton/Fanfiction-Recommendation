#!/bin/sh
#$ -N pagerank
#$ -cwd
#$ -t 1-100
python pagerank.py $SGE_TASK_ID
