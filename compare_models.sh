#!/usr/bin/env bash

python compare_models.py \
    --previous_model UHL2 \
    --our_model framewise_out_musdb_test_set/dense_models_framewise/MHMMDenseLSTM_nc4sl7ic2hs128lr0.001wd1e-06obL1ptFalselaFalseevv4mfsqrtnf512tc-1ks5us60.0ls30.0nmTruethNonenfft1025cs0.0rgFalseuvTrueif0.0of0im0.0gs0.0/test \
    --save_dir final_results_musdb_uniform_volumes
