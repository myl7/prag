#!/bin/bash

# Define the configurations arrays
declare -a configs

# Config 1: Different num_docs with top_k = 16
# p from 14 to 20 inclusive
for p in {14..20}; do
    num_docs=$((2**p))
    configs+=("NUM_DOCS=${num_docs} TOP_K=16")
done

# Config 2: num_docs = 2**17 with different top_k
# p from 0 to 6 inclusive
num_docs_fixed=$((2**17))
for p in {0..6}; do
    top_k=$((2**p))
    configs+=("NUM_DOCS=${num_docs_fixed} TOP_K=${top_k}")
done

echo "Running ${#configs[@]} configurations in batches of 7..."

# Loop through configs and run in background
counter=0
for config in "${configs[@]}"; do
    # Read config into variables
    eval "$config"

    # Assign a unique port for each job to avoid conflict in crypten/torch.distributed
    port=$((29500 + counter))
    counter=$((counter + 1))

    (
        # Set environment variables and run python script
        # MASTER_ADDR and MASTER_PORT are needed for crypten.init() / dist.init_process_group
        output=$(NUM_DOCS=$NUM_DOCS TOP_K=$TOP_K MASTER_ADDR=localhost MASTER_PORT=$port python beir_MPC_example.py 2>/dev/null)

        # Get the last line of output
        last_line=$(echo "$output" | tail -n 1)

        # Print result immediately
        echo "NUM_DOCS=$NUM_DOCS TOP_K=$TOP_K $last_line"
    ) &

    # Wait for every 7 background jobs
    if (( counter % 7 == 0 )); then
        wait
    fi
done

# Wait for all background jobs to finish
wait
