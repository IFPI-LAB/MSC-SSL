import argparse


def default_parser():
    parser = argparse.ArgumentParser()
    # Common parameter
    parser.add_argument("--data_root", "-r", default=r"..\data\1-bottom", type=str, help="/path/to/dataset")
    parser.add_argument('--labeled_folder', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default='output/1-bottom_msc_ssl')
    parser.add_argument("--num_labels", default=30, type=int, help="number of labeled data")
    parser.add_argument('--lambda_lmmd', type=float, default=1, help='data_mmd_loss_ratio')
    parser.add_argument('--lambda_level_l', type=float, default=0.5, help='data_level_loss_ratio')
    parser.add_argument('--lambda_level_u', type=float, default=0.1, help='data_level_loss_ratio')

    # dataset config
    parser.add_argument("--dataset", "-d", default="tobacco",  type=str, help="dataset name")
    parser.add_argument("--num_unlabels", default=-1, type=int, help="number of unlabeled data")  # <0, unlabeled data为所有训练集，包括了labeled data；>0，随机选取num_unlabels个未标记样本+已选择的有标记样本
    parser.add_argument("--num_workers", default=1, type=int, help="number of thread for CPU parallel")
    # optimization config
    parser.add_argument("--model", default="resnet", type=str, help="model architecture") #resner or wideresnetleaky
    parser.add_argument("--depth", default=50,  type=int, help="model depth")
    parser.add_argument("--widen_factor", default=1,  type=int, help="widen factor for wide resnet")
    parser.add_argument("--bn_momentum", default=0.001,  type=float, help="bn momentum for wide resnet")
    parser.add_argument("--l_batch_size", "-l_bs", default=64, type=int, help="mini-batch size of labeled data")
    parser.add_argument("--ul_batch_size", "-ul_bs", default=64, type=int, help="mini-batch size of unlabeled data")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--weight_decay", "-wd", default=0.0001, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd or beta_1 for adam")
    parser.add_argument('--per_epoch_steps', type=int, default=100, help='number of training images for each epoch') #  1000
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs") # iterations 1000000
    parser.add_argument("--num_cycles", default=7.9/16, type=float, help="num cycle for CosineAnnealingLR")
    parser.add_argument("--merge_one_batch", default=0 ,type=int, help="interleave of not")
    parser.add_argument("--interleave", default=1 ,type=int, help="interleave of not")
    ## LMMD loss
    parser.add_argument('--reg_warmup', type=int, default=10, )
    parser.add_argument('--reg_warmup_iter', type=int, default=100, ) # 100
    parser.add_argument('--lmmd_feat_table_l', type=int, default=128, help='feat size for mmd table') # 128
    parser.add_argument('--lmmd_feat_table_u', type=int, default=128, help='feat size for mmd table') # 128
    parser.add_argument('--lmmd_threshold', default=0.7, type=float, help='lmmd loss threshold in terms of outputs entropy')
    ## transfer learning
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--imprint', type=int, default=1,help='imprint for pretrained classifier')
    # evaluation checkpoint config
    parser.add_argument("--eval_every", default=1, type=int, help="eval every N epoches")
    parser.add_argument("--save_every", default=200, type=int, help="save every N epoches")
    parser.add_argument("--resume", default=None, type=str, help="path to checkpoint model")
    # misc
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--pretrained_weight_path", default="ckpt", type=str, help="pretrained_weight_path")

    return parser


def get_args():
    parser = default_parser()
    args = parser.parse_args()
    args.n_imgs_per_epoch = args.per_epoch_steps * args.l_batch_size
    args.iteration = args.epochs * args.per_epoch_steps
    args.net_name = f"{args.model}_{args.depth}_{args.widen_factor}"
    args.task_name = f"{args.dataset}@{args.num_labels}"

    print('default model',args.model,args.depth,args.widen_factor)
    return args

