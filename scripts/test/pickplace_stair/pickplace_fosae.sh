export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 5
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env pickplace_stair --approach fosae_policy \
        --seed $seed --offline_data_method "demo" \
        --exclude_domain_feat "none" \
        --excluded_predicates "HandSees,ViewableArm,HoldingTgt,HoldingStair,HandEmpty,Reachable,Near,OnGround,OnStair" \
        --num_train_tasks 2000 \
        --load_data \
        --timeout 30 \
        --fosae_max_n 5 \
        --spot_graph_nav_map "sqh_final" \
        --gnn_option_policy_solve_with_shooting True \
        --fosae_pred_config "predicators/config/pickplace_stair/fosae.yaml" \
        --approach_dir "saved_approaches/final/pickplace_stair/fosae_policy_$seed" \
        --ivntr_nsrt_path saved_approaches/final/pickplace_stair/ivntr_$seed/pickplace_stair__ivntr__${seed}__HandSees,ViewableArm,HoldingTgt,HoldingStair,HandEmpty,Reachable,Near,OnGround,OnStair___aesuperv_False__.saved.neupi_info \
        --log_file logs/final/pickplace_stair/sim/fosae_policy_ood_$seed.log; then
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
    if python3 predicators/main.py --env pickplace_stair --approach fosae_policy \
        --seed $seed --offline_data_method "demo" \
        --exclude_domain_feat "none" \
        --excluded_predicates "HandSees,ViewableArm,HoldingTgt,HoldingStair,HandEmpty,Reachable,Near,OnGround,OnStair" \
        --num_train_tasks 2000 \
        --load_data \
        --load_approach \
        --timeout 30 \
        --fosae_max_n 5 \
        --spot_graph_nav_map "sqh_final" \
        --gnn_option_policy_solve_with_shooting True \
        --in_domain_test True \
        --fosae_pred_config "predicators/config/pickplace_stair/fosae.yaml" \
        --approach_dir "saved_approaches/final/pickplace_stair/fosae_policy_$seed" \
        --ivntr_nsrt_path saved_approaches/final/pickplace_stair/ivntr_$seed/pickplace_stair__ivntr__${seed}__HandSees,ViewableArm,HoldingTgt,HoldingStair,HandEmpty,Reachable,Near,OnGround,OnStair___aesuperv_False__.saved.neupi_info \
        --log_file logs/final/pickplace_stair/sim/fosae_policy_in_domain_$seed.log; then
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