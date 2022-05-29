python3 main.py --run_id ex_1 --pooling_type max_pool2d

python3 main.py --run_id ex_2 --pooling_type generalized_lehmer_pool --alpha 1.3 --beta -1.5

python3 main.py --run_id ex_3 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1

python3 main.py --run_id ex_4 --pooling_type generalized_lehmer_pool --alpha 2.5 --beta 1.3

python3 main.py --run_id ex_5 --pooling_type generalized_power_mean_pool --gamma 1.3 --delta -1.5

python3 main.py --run_id ex_6 --pooling_type generalized_power_mean_pool --gamma 1.5 --delta 0.2

python3 main.py --run_id ex_7 --pooling_type generalized_power_mean_pool --gamma 2.5 --delta 1.3

python3 main.py --run_id ex_8 --pooling_type generalized_lehmer_pool --alpha 2.5 --beta -1.3

python3 main.py --run_id ex_9 --pooling_type generalized_lehmer_pool --alpha 1.3 --beta 1.3

python3 main.py --run_id ex_10 --pooling_type generalized_power_mean_pool --gamma 2.5 --delta -1.3

python3 main.py --run_id ex_11 --pooling_type generalized_power_mean_pool --gamma 1.3 --delta 1.3


# python3 main.py --run_id ex_1 --pooling_type max_pool2d --epochs 40

python3 main.py --run_id ex_2 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --epochs 40

python3 main.py --run_id ex_2 --pooling_type generalized_lehmer_pool --alpha 2.3 --beta 1.1 --epochs 40 --lr 1e-3
python3 main.py --run_id ex_1 --pooling_type max_pool2d --lr 1e-3 --epochs 40 --batch_size 248

python3 main.py --run_id ex_2 --pooling_type generalized_lehmer_pool --alpha 2.5 --beta 1.3 --lr 1e-3 --epochs 70 --batch_size 248

python3 main.py --run_id ex_3 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --lr 1e-3 --epochs 70 --batch_size 248

python3 main.py --run_id ex_4 --pooling_type lp_pool --norm_type 2 --lr 1e-3 --epochs 70 --batch_size 256
python3 main.py --run_id ex_4 --pooling_type lp_pool --norm_type 3 --lr 1e-3 --epochs 70 --batch_size 256
# --checkpoint_path ./.assets/checkpoints/ex_3_frzp_dconv_bn_220423-015008622710_generalized_lehmer_pool/10/checkpoint.pth


# python3 main.py --run_id ex_6_2 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --lr 1e-3 --epochs 70 --batch_size 256

python3 main.py --run_id ex_8_2_2lp_tail_mp --pooling_type lp_pool --norm_type 2 --lr 1e-3 --epochs 70 --batch_size 256

python3 main.py --run_id ex_1_2 --pooling_type max_pool2d --lr 1e-3 --epochs 40 --batch_size 256

python3 main.py --run_id ex_6_3_2glp_ht --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --lr 1e-3 --epochs 70 --batch_size 256

python3 main.py --run_id ex_8_3_2lp_ht --pooling_type lp_pool --norm_type 2 --lr 1e-3 --epochs 70 --batch_size 256
