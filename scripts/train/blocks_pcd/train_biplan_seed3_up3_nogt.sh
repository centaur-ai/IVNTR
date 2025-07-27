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
        --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear,Matched,FaceUp,FaceDown,Single" \
        --neupi_pred_config "predicators/config/blocks_engrave/pred_up3.yaml" \
        --neupi_do_normalization True \
        --load_data \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --neupi_parallel_invention False \
        --neupi_cache_input_graph True \
        --neupi_gt_ae_matrix False \
        --num_train_tasks 500 \
        --sesame_task_planner "astar" \
        --neupi_save_path "saved_approaches/blocks_engrave_pcdnorm/ivntr_$seed" \
        --log_file logs/blocks_engrave_pcdnorm/ivntr_up3_$seed.log; then
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

    # # Record start time
    # start_time=$(date +%s)

    # # blocks
    # if python3 predicators/main.py --env blocks_onclear --approach ivntr \
    #     --seed $seed --offline_data_method "demo" \
    #     --excluded_predicates "On,OnTable,GripperOpen,Holding,Clear" \
    #     --num_train_tasks 500 \
    #     --neupi_pred_config "predicators/config/blocks_onclear/pred.yaml" \
    #     --neupi_do_normalization True \
    #     --load_data \
    #     --sesame_task_planner "astar" \
    #     --neupi_save_path "saved_approaches/Repro_1108/blocks_onclear500_ivntr_$seed" >> logs/Repro_1108/blocks_onclear500_ivntr/$seed.log 2>&1; then
    #     echo "Seed $seed completed successfully."
    # else
    #     echo "Seed $seed encountered an error."
    # fi

    # # Record end time
    # end_time=$(date +%s)

    # # Calculate the duration in seconds
    # runtime=$((end_time - start_time))

    # # Convert to hours, minutes, and seconds
    # hours=$((runtime / 3600))
    # minutes=$(( (runtime % 3600) / 60 ))
    # seconds=$((runtime % 60))

    # # Output the total runtime
    # echo "blocks_onclear Seed $seed completed in: ${hours}h ${minutes}m ${seconds}s"

    # # Record start time
    # start_time=$(date +%s)

    # # satellites
    # if python3 predicators/main.py --env satellites --approach ivntr \
    #     --seed $seed --offline_data_method "demo" \
    #     --excluded_predicates "ViewClear,IsCalibrated,HasChemX,HasChemY,Sees" \
    #     --num_train_tasks 500 \
    #     --neupi_pred_config "predicators/config/satellites/pred.yaml" \
    #     --neupi_do_normalization False \
    #     --load_data \
    #     --sesame_task_planner "astar" \
    #     --neupi_save_path "saved_approaches/Repro_1108/satellites500_ivntr_$seed" >> logs/Repro_1108/satellites500_ivntr/$seed.log 2>&1; then
    #     echo "Seed $seed completed successfully."
    # else
    #     echo "Seed $seed encountered an error."
    # fi

    # # Record end time
    # end_time=$(date +%s)

    # # Calculate the duration in seconds
    # runtime=$((end_time - start_time))

    # # Convert to hours, minutes, and seconds
    # hours=$((runtime / 3600))
    # minutes=$(( (runtime % 3600) / 60 ))
    # seconds=$((runtime % 60))

    # # Output the total runtime
    # echo "satellites Seed $seed completed in: ${hours}h ${minutes}m ${seconds}s"
done
# --neupi_save_path "saved_approaches/8p_hard500_col_precon" \
# --wandb_run_name "8p_hard500_col_precon"

# --neupi_load_pretrained 'saved_approaches/GT_hard500_col'