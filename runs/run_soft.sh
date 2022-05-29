python3 main.py --run_id soft_1 --pooling_type max_pool2d --lr 1e-3 --epochs 40 --batch_size 256

python3 main.py --run_id soft_2 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --lr 1e-3 --epochs 70 --batch_size 256

python3 main.py --run_id soft_3 --pooling_type lp_pool --norm_type 2 --lr 1e-3 --epochs 70 --batch_size 256



# python3 advattack.py --run_id soft_1_adv --pooling_type max_pool2d --lr 1e-3 --batch_size 128 --checkpoint_path .assets/checkpoints/soft_1_220425-114745391385_max_pool2d/15/checkpoint.pth

# python3 advattack.py --run_id soft_2_adv --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets/checkpoints/soft_2_220425-115034941542_generalized_lehmer_pool/15/checkpoint.pth

# python3 advattack.py --run_id soft_3_adv --pooling_type lp_pool--batch_size 128 --checkpoint_path .assets/checkpoints/soft_3_220425-115505876961_lp_pool/15/checkpoint.pth
