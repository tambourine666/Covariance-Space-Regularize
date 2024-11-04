python train.py -project fact \
-dataset mini_imagenet \
-base_mode 'ft_cos' \
-new_mode 'ft_cos' \
-dataroot 'PUT YOUR DATA HERE' \
-gamma 0.1 \
-lr_base 0.1 \
-lr_new 0.001 \
-decay 0.0005 \
-epochs_base 1000 \
-schedule Cosine  \
-gpu 2 \
-temperature 16 \
-alpha 0.5 \
-balance 0.01 \
-loss_iter 150 \
-eta 0.1 \
-start_session 0 \
-epochs_new 40 \
-cov_restriction \
-incremental_cov_balance 0.01 \
-a 1.0>>cov/Mini-FACT.txt
