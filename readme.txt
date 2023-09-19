1. 基础环境
python 3.7
torch 1.10.1

2. 运行命令示例
cd SSL-MSC\code
python main.py --data_root ..\data\1-bottom --model_save_path output/1-bottom --num_labels 30 --lambda_level_l 0.5 --lambda_level_u 0.1 --lambda_lmmd 1
注：上述命令初始化随机挑选num_labels个样本作为有标记样本。

如果要使用挑选的标记样本，则使用以下命令：
python main.py --data_root ..\data\1-bottom --model_save_path output/1-bottom --labeled_folder iter_9 --num_labels 30 --lambda_level_l 0.5 --lambda_level_u 0.1 --lambda_lmmd 1

test
