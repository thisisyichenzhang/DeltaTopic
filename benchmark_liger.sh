#!/bin/bash

# Define the directory containing the models
model_dir="models"

# Store all model_ids in an array
model_ids=(${model_dir}/*v4)

# Get the length of the array
length=${#model_ids[@]}

# Run python scripts for every two model_ids
for ((i=0; i<$length; i+=2)); do
  # Extract the model_ids by removing the directory prefix and the file extension
  for ((j=0; j<2; j++)); do
    if [ $((i+j)) -lt $length ]; then
      model_id=$(basename "${model_ids[$((i+j))]}" .v4)

      # Print the model_id
      echo "Running for model_id: $model_id"
    
      # Run the benchmark_eval.py script with the current model_id
      python benchmark_eval.py --model_id "$model_id" &
    fi
  done
  
  # wait for all processes to finish before moving on to the next set
  wait
done
