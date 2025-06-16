export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach fosae_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --num_train_tasks 500 \
        --load_data \
        --fosae_pred_config predicators/config/blocks_engrave/fosae.yaml \
        --gnn_option_policy_solve_with_shooting True \
        --timeout 20 \
        --gnn_use_pointnet True \
        --fosae_max_n 7 \
        --load_approach \
        --ivntr_nsrt_path "saved_approaches/blocks_engrave_pcdnorm/ivntr_${seed}/blocks_engrave_pcdnorm__ivntr__${seed}__On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown___aesuperv_False__.saved" \
        --approach_dir "saved_approaches/final/blocks_pcd/fosae_policy_$seed" \
        --log_file logs/final/blocks_pcd/sim/fosae_policy_ood_$seed.log; then
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
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach fosae_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --num_train_tasks 500 \
        --load_data \
        --fosae_pred_config predicators/config/blocks_engrave/fosae.yaml \
        --gnn_option_policy_solve_with_shooting True \
        --timeout 20 \
        --gnn_use_pointnet True \
        --fosae_max_n 7 \
        --load_approach \
        --in_domain_test True \
        --ivntr_nsrt_path "saved_approaches/blocks_engrave_pcdnorm/ivntr_${seed}/blocks_engrave_pcdnorm__ivntr__${seed}__On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown___aesuperv_False__.saved" \
        --approach_dir "saved_approaches/final/blocks_pcd/fosae_policy_$seed" \
        --log_file logs/final/blocks_pcd/sim/fosae_policy_in_domain_$seed.log; then
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