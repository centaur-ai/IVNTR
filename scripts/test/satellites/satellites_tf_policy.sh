export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0 1 2 3 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach transformer_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --exclude_domain_feat "none" \
        --domain_sampler_data_filter "none" \
        --num_train_tasks 500 \
        --load_data \
        --gnn_layer_size 128 \
        --gnn_batch_size 512 \
        --gnn_option_policy_solve_with_shooting True \
        --timeout 5 \
        --load_approach \
        --ivntr_nsrt_path saved_approaches/final/satellites/ivntr_${seed}/satellites__ivntr__${seed}__ViewClear,IsCalibrated,HasChemX,HasChemY,Sees___aesuperv_False__.saved.neupi_info \
        --approach_dir "saved_approaches/final/satellites/tf_policy_$seed" \
        --log_file logs/final/satellites/sim/tf_policy_ood_$seed.log; then
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

    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env satellites --approach transformer_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
        --num_train_tasks 500 \
        --exclude_domain_feat "none" \
        --domain_sampler_data_filter "none" \
        --load_data \
        --gnn_layer_size 128 \
        --gnn_batch_size 512 \
        --load_approach \
        --in_domain_test True \
        --gnn_option_policy_solve_with_shooting True \
        --timeout 5 \
        --ivntr_nsrt_path saved_approaches/final/satellites/ivntr_${seed}/satellites__ivntr__${seed}__ViewClear,IsCalibrated,HasChemX,HasChemY,Sees___aesuperv_False__.saved.neupi_info \
        --approach_dir "saved_approaches/final/satellites/tf_policy_$seed" \
        --log_file logs/final/satellites/sim/tf_policy_in_domain_$seed.log; then
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