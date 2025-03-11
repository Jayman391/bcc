#!/bin/bash

sbatch run_clf_ensemble.sh

sbatch make_requests.sh

for dir in data/requests/*; do
 
    for file in $dir/*; do
        sbatch run_llm_inference.sh $file $dir
    done

# wait 72h for the llm_inference job to finish

sleep 259200

sbatch run_analytics.sh