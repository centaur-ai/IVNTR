export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 4 5
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env view_plan_hard --approach gnn_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "HandSees,ViewableArm,Calibrated,Holding,HandEmpty,Reachable,Near,Close,OnGround,OnStair" \
        --num_train_tasks 2000 \
        --load_data \
        --timeout 30 \
        --gnn_layer_size 512 \
        --gnn_batch_size 1024 \
        --gnn_option_policy_solve_with_shooting True \
        --spot_graph_nav_map "sqh_final" \
        --gnn_do_normalization True \
        --ivntr_nsrt_path saved_approaches/final/view_plan_hard/ivntr_${seed}/view_plan_hard__ivntr__${seed}__HandSees,ViewableArm,Calibrated,Holding,HandEmpty,Reachable,Near,Close,OnGround,OnStair___aesuperv_False__.saved.neupi_info \
        --approach_dir "saved_approaches/final/view_plan_hard/gnn_policy_$seed" \
        --log_file logs/final/view_plan_hard/sim/gnn_ood_$seed.log; then
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
    if python3 predicators/main.py --env view_plan_hard --approach gnn_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "HandSees,ViewableArm,Calibrated,Holding,HandEmpty,Reachable,Near,Close,OnGround,OnStair" \
        --num_train_tasks 2000 \
        --load_data \
        --gnn_layer_size 512 \
        --gnn_batch_size 1024 \
        --load_approach \
        --timeout 30 \
        --gnn_option_policy_solve_with_shooting True \
        --in_domain_test True \
        --gnn_do_normalization True \
        --spot_graph_nav_map "sqh_final" \
        --ivntr_nsrt_path saved_approaches/final/view_plan_hard/ivntr_${seed}/view_plan_hard__ivntr__${seed}__HandSees,ViewableArm,Calibrated,Holding,HandEmpty,Reachable,Near,Close,OnGround,OnStair___aesuperv_False__.saved.neupi_info \
        --approach_dir "saved_approaches/final/view_plan_hard/gnn_policy_$seed" \
        --log_file logs/final/view_plan_hard/sim/gnn_in_domain_$seed.log; then
        echo "Seed $seed completed successfully."
    else
        echo "Seed $seed encountered an error."
    fi

done