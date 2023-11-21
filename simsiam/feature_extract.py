import argparse
import os
import numpy as np

from tobacco_dataset import TBCdataset
from model.builder import gen_model

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# args
parser = argparse.ArgumentParser(description='PyTorch Infer')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_classes', default=3, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--model_path', default=None, type=str,
                    help='path to checkpoint')
parser.add_argument('--data_path', default=None, type=str,
                    help='data path')
parser.add_argument('--save_npz', default='feature.npz', type=str,
                    help='feature save path')


def get_embedding(args):
    # select device
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        print("CUDA is NOT available")
        device = "cpu"

    # create model
    model = gen_model(depth=50, num_classes=args.num_classes)

    # rename simsiam pre-trained keys
    checkpoint = torch.load(args.model_path)  # load simsiam model
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
            # remove prefix
            state_dict[k[len("module.encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # prepare data
    tbc_dataset = TBCdataset(args.data_path)
    loader = torch.utils.data.DataLoader(tbc_dataset, batch_size=1, shuffle=False,
                                         num_workers=0, pin_memory=True)

    # infer
    with torch.no_grad():
        forward_fn = model.forward
        embedding_all = None
        label_all = None
        im_path_all = []
        for i, (images, targets, im_path) in enumerate(loader):
            images = images.cuda()

            out_lists_inter = forward_fn(images, return_fmap=True)

            embedding = out_lists_inter[-2]
            embedding = embedding.detach().cpu().numpy()

            label = targets.cpu().numpy()

            im_path_all.append(im_path)

            try:
                embedding_all = np.append(embedding_all, embedding, axis=0)
            except:
                embedding_all = embedding

            try:
                label_all = np.append(label_all, label, axis=0)
            except:
                label_all = label

    return embedding_all, label_all, im_path_all


if __name__ == "__main__":
    args = parser.parse_args()

    # get embedding
    embedding_all, label_all, im_path_all = get_embedding(args)
    np.savez(os.path.join('embed', args.save_npz), embedding_all=embedding_all, label_all=label_all,
             im_path_all=im_path_all)
