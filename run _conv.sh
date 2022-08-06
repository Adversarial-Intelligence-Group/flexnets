python3 main.py --run_id ex_glc_lb

python3 advattack.py --run_id ex_glc_lb --batch_size 128 --checkpoint_path .assets/checkpoints/ex_glc_lb_220629-165244570400_max_pool2d/best_model.pth

python3 main.py --run_id ex_gpc_lb

python3 advattack.py --run_id ex_glc_lb --batch_size 128 --checkpoint_path .assets/checkpoints/ex_gpc_lb_220629-171209420735_max_pool2d/best_model.pth

python3 main.py --run_id ex_gpc_fb

python3 advattack.py --run_id ex_gpc_fb --batch_size 128 --checkpoint_path .assets/checkpoints/ex_gpc_fb_220629-182428193689_max_pool2d/checkpoint.pth

python3 main.py --run_id ex_gln

python3 main.py --run_id ex_gln_train

python3 main.py --run_id ex_gln_train_beta

python3 advattack.py --run_id ex_gln_train --batch_size 128 --checkpoint_path .assets/checkpoints/ex_gln_train_220629-211314608394_max_pool2d/best_model.pth

python3 main.py --run_id ex_lb_glc_gln

python3 advattack.py --run_id ex_gln_train --batch_size 128 --checkpoint_path .assets/checkpoints/ex_lb_glc_gln_220629-213651864111_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_lb_glc_gln --batch_size 128 --checkpoint_path .assets/checkpoints/ex_lb_glc_gln_220629-213651864111_max_pool2d/best_model.pth


python3 main.py --run_id ex_lb_gpc_gln

python3 advattack.py --run_id ex_lb_gpc_gln --batch_size 128 --checkpoint_path .assets/checkpoints/ex_lb_gpc_gln_220629-222413366740_max_pool2d/best_model.pth

python3 main.py --run_id ex_glc_mid

python3 advattack.py --run_id ex_glc_mid --batch_size 128 --checkpoint_path .assets/checkpoints/ex_glc_mid_220629-225135742395_max_pool2d/best_model.pth

python3 main.py --run_id ex_gpc_mid

python3 advattack.py --run_id ex_gpc_mid --batch_size 128 --checkpoint_path .assets/checkpoints/ex_gpc_mid_220630-011730406135_max_pool2d/best_model.pth

python3 main.py --run_id ex_glc_lb_net --pooling_type generalized_lehmer_pool --alpha 1.8 --beta 1.3

python3 advattack.py --run_id ex_glc_lb_net --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets/checkpoints/ex_glc__lb_net_220630-104546868915_generalized_lehmer_pool/best_model.pth

python3 advattack.py --run_id ex_glc_lb_net --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets_last_nets/checkpoints/ex_glc_lb_net_220630-112036920670_generalized_lehmer_pool/best_model.pth

python3 main.py --run_id ex_gpc_lb_net --pooling_type generalized_lehmer_pool --alpha 1.8 --beta 1.3

python3 advattack.py --run_id ex_gpc_lb_net --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets_last_nets/checkpoints/ex_gpc_lb_net_220630-114521913464_generalized_lehmer_pool/best_model.pth



python3 advattack.py --run_id ex_gpc_glp_lb --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets_net/checkpoints/ex_gpc_net_lb_220629-002457813090_max_pool2d/best_model.pth




python3 advattack.py --run_id ex_gln_train --batch_size 128 --checkpoint_path .assets_last_nets/checkpoints/ex_gln_train_220629-211314608394_max_pool2d/best_model.pth


















