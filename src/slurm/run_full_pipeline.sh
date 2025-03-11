#!/bin/bash

sbatch run_clf_ensemble.sh

sbatch make_requests.sh

for file in data/requests/*; do
 
    sbatch run_llm_inference.sh $file

# wait 72h for the llm_inference job to finish

sleep 259200

sbatch run_analytics.sh