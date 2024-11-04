python train.py -project base \
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
-gpu 3 \
-temperature 16 \
-alpha 0.5 \
-balance 0.01 \
-eta 0.1 \
-start_session 0 \
-cov_restriction \
-epochs_new 20 \
-cov_balance 0.01 \
-incremental_cov_balance 0.01 \
-a 1.0 >>MINI_CE.txt



