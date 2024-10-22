# step1: initialize image bank
python datafree_kd.py \
--method cmi \
--dataset cifar10 \
--batch_size 128 \
--synthesis_batch_size 200 \
--teacher wrn40_2 \
--student wrn16_1 \
--lr 0.1 \
--g_steps 500 \
--lr_g 1e-3 \
--epochs 400 \
--adv 0 \
--bn 1 \
--oh 1 \
--cr 0.8 \
--cr_T 0.2 \
--gpu 1 \
--seed 1 \
--save_dir run/cmi/cifar10_wrn402_initBank \
--log_tag init_bank 

# step2: train student with cmi
python datafree_kd.py \
--method cmi \
--dataset cifar10 \
--batch_size 128 \
--synthesis_batch_size 200 \
--teacher wrn40_2 \
--student wrn16_1 \
--lr 0.1 \
--kd_steps 2000 \
--ep_steps 2000 \
--g_steps 500 \
--lr_g 1e-3 \
--epochs 100 \
--adv 0.5 \
--bn 1 \
--oh 1 \
--cr 0.8 \
--cr_T 0.2 \
--T 20 \
--gpu 1 \
--seed 1 \
--save_dir run/cmi/cifar10_wrn402_wrn161 \
--init_bank run/cmi/cifar10_wrn402_initBank \
--run_kd

