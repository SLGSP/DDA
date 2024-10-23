python datafree_kd.py \
--method adi \
--dataset cifar10 \
--batch_size 256 \
--teacher wrn40_2 \
--student wrn16_1 \
--lr 0.1 \
--epochs 200 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 1000 \
--lr_g 0.1 \
--adv 1 \
--bn 10 \
--oh 1 \
--T 20 \
--act 0 \
--balance 0 \
--gpu 4 \
--seed 1 \
--save_dir run/adi \
--run_kd