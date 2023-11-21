import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy, random, time, json
import queue
import torch
import torch.optim as optim

from ssl_lib.models.builder import gen_model
from ssl_lib.datasets.builder import gen_dataloader
from ssl_lib.param_scheduler import scheduler
from ssl_lib.utils import Logger
from ssl_lib.trainer.train import train, evaluate
from ssl_lib.trainer.imprint import imprint
from ssl_lib.consistency.focal_loss import focal_loss


def get_file_quantity(folder: str) -> int:
    '''Get the total number of files in the folder'''
    # Determine the initial folder
    assert os.path.isdir(folder), 'Please enter valid folder parameters'
    file_quantity = 0
    folder_path_queue = queue.Queue()
    folder_path_queue.put_nowait(folder)
    # Processing folder in the queue
    while not folder_path_queue.empty():
        folder = folder_path_queue.get_nowait()
        file_folder_list = list(map(lambda bar: os.path.join(folder, bar), os.listdir(folder)))
        folder_list = list(filter(lambda bar: os.path.isdir(bar), file_folder_list))
        for folder_path in folder_list:
            folder_path_queue.put_nowait(folder_path)
        temp_file_count = len(file_folder_list) - len(folder_list)
        file_quantity += temp_file_count
    return file_quantity


def main(cfg):
    # set seed
    random.seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
    else:
        print("CUDA is NOT available")
        device = "cpu"

    # build data loader
    print("load dataset")
    lt_loader, ult_loader, test_loader, num_classes, img_size, num_labeled = gen_dataloader(cfg.data_root, cfg.dataset, cfg=cfg)
    cfg.num_classes = num_classes

    # build model
    if not os.path.exists(cfg.model_save_path):
        os.makedirs(cfg.model_save_path)
    model = gen_model(cfg.model, cfg.depth, cfg.widen_factor, num_classes, cfg.pretrained_weight_path, cfg.pretrained,
                      bn_momentum=cfg.bn_momentum).to(device)
    if cfg.imprint:
        model = imprint(model, lt_loader, num_classes, cfg.num_labels, device)

    model = torch.nn.DataParallel(model)
    model.train()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(model)

    # build optimizer
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params, 'weight_decay': cfg.weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]

    optimizer = optim.SGD(param_list, lr=cfg.lr, momentum=cfg.momentum, weight_decay=0, nesterov=True)

    # set lr scheduler
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer, cfg.iteration, num_cycles=cfg.num_cycles)

    # init meter
    start_epoch = 0
    log_names = ['Epoch', 'Learning Rate', 'Train Loss', 'Loss CE', 'Loss LMMD', 'Loss MSC L', 'Loss MSC U',
                 'Labeled Acc', 'Unlabeled Acc', 'Mask LMMD', 'Test Loss', 'Test Acc.', 'Time']

    if cfg.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(cfg.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(cfg.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger = Logger(cfg.model_save_path, title=cfg.task_name, resume=True)
    else:
        logger = Logger(cfg.model_save_path, title=cfg.task_name)
        logger.set_names(log_names)

    print("training")
    test_acc_list = []
    time_record = time.time()
    best_acc = 0

    for epoch in range(start_epoch, cfg.epochs):
        lr = optimizer.param_groups[0]['lr']

        train_loader = zip(lt_loader, ult_loader)
        train_logs = train(epoch, train_loader, model, optimizer, lr_scheduler, cfg, device)
        dtime = time.time() - time_record

        test_loss, test_acc = evaluate(model, test_loader, device)
        test_acc_list.append(test_acc)
        logger.append((epoch, lr) + train_logs + (test_loss, test_acc, dtime))

        if (epoch + 1) % cfg.save_every == 0:
            filepath = os.path.join(cfg.save_path, f'{cfg.net_name}_{epoch + 1}.pth')
            torch.save({'epoch': epoch + 1,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()}, filepath)

        time_record = time.time()

        # save model
        if best_acc < test_acc:
            best_acc = test_acc
            if cfg.model_save_path != '':
                if not os.path.exists(cfg.model_save_path):
                    os.makedirs(cfg.model_save_path)
                torch.save(model.state_dict(), os.path.join(cfg.model_save_path, 'best_model.pth'))

    accuracies = {}
    logger.write("best test acc: {}".format(best_acc))
    for i in [1, 10, 20, 50]:
        logger.write(f"mean test acc. over last {i} checkpoints: {numpy.mean(test_acc_list[-i:])}")
        logger.write(f"median test acc. over last {i} checkpoints: {numpy.median(test_acc_list[-i:])}")
        accuracies[f"mean_last{i}"] = numpy.mean(test_acc_list[-i:])
        accuracies[f"mid_last{i}"] = numpy.median(test_acc_list[-i:])
    logger.close()


if __name__ == "__main__":
    from parser1 import get_args
    args = get_args()
    if args.labeled_folder is not None:
        args.num_labels = get_file_quantity(os.path.join(args.data_root, 'labeled', args.labeled_folder))
    print('args:', args)
    main(args)
