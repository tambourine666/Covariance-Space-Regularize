python train.py -project base \
-dataset cifar100  \
-dataroot 'PUT YOUR DATA HERE' \
-base_mode "ft_cos" \
-new_mode "ft_cos" \
-gamma 0.1 \
-lr_base 0.1 \
-lr_new 0.001 \
-decay 0.0005 \
-epochs_base 800 \
-schedule Cosine \
-gpu 1 \
-temperature 16 \
-batch_size_base 256 \
-balance 0.001 \
-loss_iter 0 \
-start_session 0 \
-alpha 0.5  \
-cov_restriction \
-cov_balance 0.01 \
-a 1.0 \
-incremental_cov_balance 0.01 \
-epochs_new 30 >>cifar_CE.txt





