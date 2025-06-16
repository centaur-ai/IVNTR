export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 1
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env view_plan_trivial --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --exclude_domain_feat "none" \
        --excluded_predicates "HandSees,ViewClear,Viewable,Calibrated" \
        --neupi_pred_config "predicators/config/view_plan_trivial/pred.yaml" \
        --neupi_do_normalization True \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "fdsat" \
        --num_train_tasks 500 \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --neupi_equ_dataset 0.05 \
        --neupi_pred_search_dataset 1.0 \
        --bilevel_plan_without_sim False \
        --execution_monitor expected_atoms \
        --spot_graph_nav_map "debug" \
        --domain_aaai_thresh 300000 \
        --approach_dir "saved_approaches/final/view_plan_trivial/ivntr_$seed" \
        --neupi_save_path "saved_approaches/final/view_plan_trivial/ivntr_$seed" \
        --log_file logs/final/view_plan_trivial/sim/ivntr_relearn_ood_$seed.log; then
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