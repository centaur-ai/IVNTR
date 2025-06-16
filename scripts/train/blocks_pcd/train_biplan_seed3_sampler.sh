export FD_EXEC_PATH="ext/downward"
export PYTHONHASHSEED=0

for seed in 3
do
    echo "Running Seed $seed --------------------------------------"

    # Record start time
    start_time=$(date +%s)

    # view_pan
    if python3 predicators/main.py --env blocks_engrave_pcdnorm --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Single,Matched,FaceUp,FaceDown" \
        --neupi_pred_config "predicators/config/blocks_engrave/pred_all.yaml" \
        --neupi_do_normalization True \
        --exclude_domain_feat "none" \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --load_data \
        --load_neupi_from_json True \
        --neupi_parallel_invention False \
        --neupi_cache_input_graph True \
        --neupi_gt_ae_matrix False \
        --num_train_tasks 500 \
        --sesame_task_planner "fdsat" \
        --neupi_learning_dataset 1.0 \
        --domain_aaai_thresh 500000 \
        --timeout 30 \
        --approach_dir "saved_approaches/blocks_engrave_pcdnorm/ivntr_$seed" \
        --log_file logs/final/blocks_pcd/ivntr_ood_$seed.log; then
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
    echo "view_plan_hard Seed $seed completed in: ${hours}h ${minutes}m ${seconds}s"

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
        --timeout 10 \
        --in_domain_test True \
        --approach_dir "saved_approaches/blocks_engrave_pcdnorm/ivntr_$seed" \
        --log_file logs/final/blocks_pcd/ivntr_indomain_$seed.log; then
        echo "Seed $seed completed successfully."
    else
        echo "Seed $seed encountered an error."
    fi
 
done
# --neupi_save_path "saved_approaches/8p_hard500_col_precon" \
# --wandb_run_name "8p_hard500_col_precon"

# --neupi_load_pretrained 'saved_approaches/GT_hard500_col'