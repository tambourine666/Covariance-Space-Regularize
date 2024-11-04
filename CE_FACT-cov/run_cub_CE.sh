python train.py -project base \
-dataroot 'PUT YOUR DATA HERE' \
-dataset cub200 \
-base_mode 'ft_cos' \
-new_mode 'ft_cos' \
-gamma 0.25 \
-lr_base 0.005 \
-lr_new 0.005 \
-decay 0.0005 \
-schedule Milestone \
-epochs_base 400 \
-milestones 200 300 \
-gpu 2 \
-temperature 16 \
-batch_size_base 256 \
-balance 0.01 \
-loss_iter 0 \
-cov_restriction \
-cov_balance 0.01 \
-epochs_new 50 \
-incremental_cov_balance 0.01 \
-a 1.0 >>cov/CUB_CE_0.01_0.01.txt


