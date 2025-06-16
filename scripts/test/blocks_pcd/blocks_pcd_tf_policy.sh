export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 2 3 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach transformer_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --num_train_tasks 500 \
        --load_data \
        --gnn_layer_size 128 \
        --gnn_batch_size 64 \
        --load_approach \
        --gnn_option_policy_solve_with_shooting True \
        --timeout 20 \
        --gnn_use_pointnet True \
        --gnn_do_normalization True \
        --approach_dir "saved_approaches/final/blocks_pcd/tf_policy_ood_$seed" \
        --ivntr_nsrt_path "saved_approaches/blocks_engrave_pcdnorm/ivntr_${seed}/blocks_engrave_pcdnorm__ivntr__${seed}__On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown___aesuperv_False__.saved" \
        --log_file logs/final/blocks_pcd/sim/tf_policy_ood_$seed.log; then
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
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach transformer_nsrt_policy \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --num_train_tasks 500 \
        --load_data \
        --gnn_layer_size 128 \
        --gnn_batch_size 16 \
        --gnn_do_normalization True \
        --gnn_option_policy_solve_with_shooting True \
        --load_approach \
        --timeout 20 \
        --in_domain_test True \
        --gnn_use_pointnet True \
        --ivntr_nsrt_path "saved_approaches/blocks_engrave_pcdnorm/ivntr_${seed}/blocks_engrave_pcdnorm__ivntr__${seed}__On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown___aesuperv_False__.saved" \
        --approach_dir "saved_approaches/final/blocks_pcd/tf_policy_ood_$seed" \
        --log_file logs/final/blocks_pcd/sim/tf_policy_in_domain_$seed.log; then
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