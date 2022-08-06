python3 advattack.py --run_id ex_1 --pooling_type max_pool2d --batch_size 128 --checkpoint_path .assets/checkpoints/ex_1_220423-103907610787_max_pool2d/13/checkpoint.pth

python3 advattack.py --run_id ex_2 --pooling_type generalized_lehmer_pool --alpha 2.5 --beta 1.3 --batch_size 128 --checkpoint_path .assets/checkpoints/ex_2_220423-110658766082_generalized_lehmer_pool/13/checkpoint.pth

python3 advattack.py --run_id ex_3 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --batch_size 128 --checkpoint_path .assets/checkpoints/ex_3_220423-111627596623_generalized_lehmer_pool/13/checkpoint.pth

python3 advattack.py --run_id ex_4 --pooling_type lp_pool --batch_size 128 --checkpoint_path .assets/checkpoints/ex_4_220424-005201627045_lp_pool/14/checkpoint.pth

python3 advattack.py --run_id ex_4 --pooling_type lp_pool --batch_size 128 --checkpoint_path .assets/checkpoints/ex_4_220424-234056554986_lp_pool/12/checkpoint.pth

python3 advattack.py --run_id ex_1 --pooling_type max_pool2d --batch_size 128 --checkpoint_path .assets/checkpoints/ex_1_2_220425-024101907121_max_pool2d/12/checkpoint.pth

python3 advattack.py --run_id ex_3 --pooling_type generalized_lehmer_pool --alpha 1.5 --beta 0.1 --batch_size 128 --checkpoint_path .assets/checkpoints/ex_3_220423-111627596623_generalized_lehmer_pool/12/checkpoint.pth
