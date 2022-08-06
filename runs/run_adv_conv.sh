python3 advattack.py --run_id ex_1_c --batch_size 128 --checkpoint_path .assets_conv/checkpoints/ex_1_c_220531-145946942239_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_3b_l --conv_type generalized_lehmer_conv --alpha 2.3 --beta -2 --batch_size 128 --checkpoint_path .assets_conv/checkpoints/ex_3b_l_220531-181123927331_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_4b_l --conv_type generalized_lehmer_conv --alpha 1.8 --beta 1.3 --batch_size 128 --checkpoint_path .assets_conv/checkpoints/ex_4b_l_220531-184827712671_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_4d_p --conv_type generalized_power_conv --gamma 1.8 --delta 1.3 --batch_size 128 --checkpoint_path .assets_conv/checkpoints/ex_4d_p_220601-123415414683_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_3d_p --conv_type generalized_power_conv --gamma 2.3 --delta 0.5 --batch_size 128 --checkpoint_path .assets_conv/checkpoints/ex_3d_p_220601-115849164362_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_gc_gp_net2 --conv_type generalized_lehmer_conv --alpha 1.5 --beta 0.1 --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets_net/checkpoints/ex_gc_gp_net2_220628-130524896396_generalized_lehmer_pool/best_model.pth

python3 advattack.py --run_id ex_net3 --conv_type generalized_power_conv --pooling_type generalized_lehmer_pool --batch_size 128 --checkpoint_path .assets_act/checkpoints/ex_10_net_220622-163012523419_generalized_lehmer_pool/best_model.pth

python3 advattack.py --run_id ex_net4 --conv_type generalized_lehmer_conv --pooling_type generalized_lehmer_pool --batch_size 64 --checkpoint_path .assets_act/checkpoints/ex_9_net_220622-143842280127_generalized_lehmer_pool/checkpoint.pth

python3 advattack.py --run_id ex_net5 --batch_size 128 --checkpoint_path .assets_act/checkpoints/ex_13_net_220622-194214566333_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_gpc_net_lb --batch_size 128 --checkpoint_path .assets_net/checkpoints/ex_gpc_net_lb_220629-002457813090_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_4_relu --batch_size 128 --checkpoint_path .assets_act/checkpoints/ex_4_relu_220622-141734135682_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_glc_wor_net_lb --batch_size 128 --checkpoint_path .assets_net/checkpoints/ex_glc_wor_net_lb_220629-011031217160_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_glc_wor_net_lb2 --batch_size 128 --checkpoint_path .assets_net/checkpoints/ex_glc_wor_net_lb2_220629-014446409726_max_pool2d/best_model.pth

python3 advattack.py --run_id ex_glc1_net_fb --batch_size 128 --checkpoint_path .assets_net/checkpoints/ex_glc1_net_fb_220629-111952653166_max_pool2d/best_model.pth










