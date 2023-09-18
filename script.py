import os


print("*************************************************************************************************************")
print("bottom:SSL + pretrained ***********************************************************************")
os.system(r"python main.py "
          r"--data_root ../data/1-bottom "
          r"--model_save_path output/device_512_9-1/bottom_semi_supervised_data_msc_v4_l_0.5_u_0.1_iter_0 "
          r"--labeled_folder iter_9 "
          r"--num_labels -1 "
          r"--lambda_level_l 0.5 "
          r"--lambda_level_u 0.1 "
          r"--lambda_lmmd 1 ")