export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0 1 2 3 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach oracle \
        --seed $seed --offline_data_method "demo" \
        --timeout 50 \
        --num_train_tasks 50 \
        --sesame_task_planner 'fdopt' \
        --log_file logs/blocks_engrave_pcdnorm/data_collect500_$seed.log; then
        echo "Seed $seed completed successfully."
    else
        echo "Seed $seed encountered an error."
    fi

    # Record end time
    end_time=$(date +%s)

    # Calculate the duration in seconds
    runtime=$((end_time - start_time))

    # Convert to hours, minutes, and seconds
    hours=$((runtime / 3600))
    minutes=$(( (runtime % 3600) / 60 ))
    seconds=$((runtime % 60))

    # Output the total runtime
    echo "Seed $seed completed in: ${hours}h ${minutes}m ${seconds}s"
done