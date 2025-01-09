python  main.py \
            --gpu           $1    \
            --eval_epoch    1  \
              --only_test     0   \
            --model_name    MEAformer \
            --data_choice   $2 \
            --data_split    $3 \
            --data_rate     $4 \
            --epoch         500 \
            --lr            5e-4  \
            --hidden_units  "300,300,300" \
            --save_model    0 \
            --batch_size    3500 \
	        --csls          \
	        --csls_k        3 \
	        --random_seed   42 \
            --exp_name      IJCAI_MEAformer_sf_$5_500-Norm \
            --exp_id        v1_$3_$4 \
            --workers       12 \
            --dist          0 \
            --accumulation_steps 1 \
            --scheduler     cos \
            --attr_dim      300     \
            --img_dim       300     \
            --name_dim      300     \
            --char_dim      300     \
            --hidden_size   300     \
            --tau           0.1     \
            --structure_encoder "gat" \
            --num_attention_heads 1 \
            --num_hidden_layers 1 \
            --use_surface   $5     \
            --use_intermediate 1   \
            --enable_sota \
            --replay 1 \
            --device cpu



