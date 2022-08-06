python3 main.py --run_id ex_1 --pooling_type max_pool2d

python3 main.py --run_id ex_2_dconv --pooling_type max_pool2d --epochs 35 --batch_size 248

python3 main.py --run_id ex_2_dconv_bn --pooling_type generalized_lehmer_pool --alpha 2.5 --beta 1.3 --epochs 150 --batch_size 248

python3 main.py --run_id ex_4_frzp_dconv_bn --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --epochs 10 --batch_size 248

python3 main.py --run_id ex_3_frzp_dconv_bn2 --pooling_type generalized_lehmer_pool --epochs 25 --batch_size 248 --checkpoint_path ./.assets/checkpoints/ex_3_frzp_dconv_bn_220423-015008622710_generalized_lehmer_pool/10/checkpoint.pth

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

python3 main.py --run_id ex_12 --pooling_type generalized_power_mean_pool --gamma 2.5 --delta 1.3 --epochs 50

python3 main.py --run_id ex_13_relu --pooling_type generalized_lehmer_pool --alpha 1.3 --beta -1.5
