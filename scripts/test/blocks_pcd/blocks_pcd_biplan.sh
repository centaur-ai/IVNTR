export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 2 3 4
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --neupi_pred_config "predicators/config/blocks_engrave/pred_all.yaml" \
        --neupi_do_normalization True \
        --exclude_domain_feat "none" \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --load_data \
        --load_approach \
        --load_neupi_from_json True \
        --neupi_parallel_invention False \
        --neupi_cache_input_graph True \
        --neupi_gt_ae_matrix False \
        --num_train_tasks 500 \
        --sesame_task_planner "fdsat" \
        --neupi_learning_dataset 1.0 \
        --domain_aaai_thresh 500000 \
        --timeout 20 \
        --approach_dir "saved_approaches/blocks_engrave_pcdnorm/ivntr_$seed" \
        --log_file logs/final/blocks_pcd/sim/ivntr_ood_$seed.log; then
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
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --neupi_pred_config "predicators/config/blocks_engrave/pred_all.yaml" \
        --neupi_do_normalization True \
        --exclude_domain_feat "none" \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --load_data \
        --load_approach \
        --load_neupi_from_json True \
        --neupi_parallel_invention False \
        --neupi_cache_input_graph True \
        --neupi_gt_ae_matrix False \
        --num_train_tasks 500 \
        --sesame_task_planner "fdsat" \
        --neupi_learning_dataset 1.0 \
        --domain_aaai_thresh 500000 \
        --timeout 20 \
        --in_domain_test True \
        --approach_dir "saved_approaches/blocks_engrave_pcdnorm/ivntr_$seed" \
        --log_file logs/final/blocks_pcd/sim/ivntr_indomain_$seed.log; then
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