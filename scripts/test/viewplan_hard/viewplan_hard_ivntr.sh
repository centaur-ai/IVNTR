export FD_EXEC_PATH=ext/downward
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 0
do
    echo "Running Seed $seed --------------------------------------"
    # Record start time
    start_time=$(date +%s)
    # low-level sampling is very hard for this environment
    if python3 predicators/main.py --env view_plan_hard --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --exclude_domain_feat "none" \
        --excluded_predicates "HandSees,ViewableArm,Calibrated,Holding,HandEmpty,Reachable,Near,Close,OnGround,OnStair" \
        --neupi_pred_config "predicators/config/view_plan_hard/pred_all.yaml" \
        --neupi_do_normalization True \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "astar" \
        --num_train_tasks 2000 \
        --load_approach \
        --load_neupi_from_json True \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --bilevel_plan_without_sim False \
        --sesame_max_samples_per_step 50 \
        --timeout 30 \
        --spot_graph_nav_map "sqh_final" \
        --neupi_load_pretrained "saved_approaches/open_models/view_plan_hard/ivntr_$seed" \
        --approach_dir "saved_approaches/open_models/view_plan_hard/ivntr_$seed" \
        --log_file logs/final/view_plan_hard/sim/ivntr_ood_$seed.log; then
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
    if python3 predicators/main.py --env view_plan_hard --approach ivntr \
        --seed $seed --offline_data_method "demo" \
        --disable_harmlessness_check True \
        --excluded_predicates "HandSees,ViewableArm,Calibrated,Holding,HandEmpty,Reachable,Near,Close,OnGround,OnStair" \
        --neupi_pred_config "predicators/config/view_plan_hard/pred_all.yaml" \
        --neupi_do_normalization True \
        --neupi_gt_ae_matrix False \
        --sesame_task_planner "astar" \
        --num_train_tasks 2000 \
        --load_data \
        --load_approach \
        --load_neupi_from_json True \
        --neupi_entropy_w 0.0 \
        --neupi_loss_w 1.0 \
        --bilevel_plan_without_sim False \
        --sesame_max_samples_per_step 50 \
        --timeout 30 \
        --in_domain_test True \
        --spot_graph_nav_map "sqh_final" \
        --neupi_load_pretrained "saved_approaches/open_models/view_plan_hard/ivntr_$seed" \
        --approach_dir "saved_approaches/open_models/view_plan_hard/ivntr_$seed" \
        --log_file logs/final/view_plan_hard/sim/ivntr_in_domain_$seed.log; then
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