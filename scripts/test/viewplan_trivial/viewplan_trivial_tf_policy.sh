export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env view_plan_trivial --approach transformer_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "HandSees,ViewClear,Viewable,Calibrated" \
        --num_train_tasks 500 \
        --load_data \
        --exclude_domain_feat "none" \
        --domain_sampler_data_filter "none" \
        --gnn_layer_size 256 \
        --gnn_batch_size 512 \
        --gnn_option_policy_solve_with_shooting True \
        --gnn_do_normalization True \
        --spot_graph_nav_map "debug" \
        --load_approach \
        --ivntr_nsrt_path saved_approaches/final/view_plan_trivial/ivntr_${seed}/view_plan_trivial__ivntr__${seed}__HandSees,ViewClear,Viewable,Calibrated___aesuperv_False__.saved.neupi_info \
        --approach_dir "saved_approaches/final/view_plan_trivial/tf_policy_$seed" \
        --log_file logs/final/view_plan_trivial/sim/tf_policy_ood_$seed.log; then
        echo "Seed $seed completed successfully."
    else
        echo "Seed $seed encountered an error."
    fi
done

for seed in 1
do
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env view_plan_trivial --approach transformer_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "HandSees,ViewClear,Viewable,Calibrated" \
        --num_train_tasks 500 \
        --load_data \
        --exclude_domain_feat "none" \
        --domain_sampler_data_filter "none" \
        --gnn_layer_size 256 \
        --gnn_batch_size 512 \
        --gnn_do_normalization True \
        --load_approach \
        --in_domain_test True \
        --gnn_option_policy_solve_with_shooting True \
        --spot_graph_nav_map "debug" \
        --ivntr_nsrt_path saved_approaches/final/view_plan_trivial/ivntr_${seed}/view_plan_trivial__ivntr__${seed}__HandSees,ViewClear,Viewable,Calibrated___aesuperv_False__.saved.neupi_info \
        --approach_dir "saved_approaches/final/view_plan_trivial/tf_policy_$seed" \
        --log_file logs/final/view_plan_trivial/sim/tf_policy_in_domain_$seed.log; then
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