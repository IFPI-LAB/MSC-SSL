
import os, numpy, random, time, math
import torch
import torch.nn.functional as F

from ssl_lib.consistency.regularizer import Distribution_Loss
from ssl_lib.param_scheduler import scheduler
from ssl_lib.utils import Bar, AverageMeter
from .supervised import supervised_train


LABELED_FEAT_TABLES=None
UNLABELED_FEAT_TABLES=None
LABELED_INPUT_TABLES=None
UNLABELED_INPUT_TABLES=None
LABELS_TABLES=None
UNLABELED_LOG_TABLES=None

def get_mask(logits,threshold, num_class=10):
    ent = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    threshold = threshold * math.log(num_class)
    mask = ent.le(threshold).float()
    return mask

def update_feat_table(cur_feat_l,cur_feat_u,cur_inputs_l,cur_inputs_u_w,cur_labels,cur_logits_u_w,feat_table_size_l=-1,feat_table_size_u=-1,mask_l=None, mask_u=None):
    global LABELED_FEAT_TABLES,UNLABELED_FEAT_TABLES,LABELED_INPUT_TABLES,UNLABELED_INPUT_TABLES,LABELS_TABLES,UNLABELED_LOG_TABLES
    if mask_l is not None:
        mask_l = mask_l.nonzero().flatten()
        mask_u = mask_u.nonzero().flatten()
        cur_feat_l=cur_feat_l[mask_l]
        cur_feat_u=cur_feat_u[mask_u]
        cur_inputs_l=cur_inputs_l[mask_l]
        cur_inputs_u_w=cur_inputs_u_w[mask_u]
        cur_labels=cur_labels[mask_l]
        cur_logits_u_w=cur_logits_u_w[mask_u]
    if feat_table_size_l>0:
        if LABELED_FEAT_TABLES is None:
            LABELED_FEAT_TABLES = cur_feat_l
            UNLABELED_FEAT_TABLES = cur_feat_u
            LABELED_INPUT_TABLES = cur_inputs_l
            UNLABELED_INPUT_TABLES = cur_inputs_u_w
            LABELS_TABLES = cur_labels
            UNLABELED_LOG_TABLES = cur_logits_u_w
        else:
            LABELED_FEAT_TABLES = torch.cat([LABELED_FEAT_TABLES,cur_feat_l])
            UNLABELED_FEAT_TABLES = torch.cat([UNLABELED_FEAT_TABLES,cur_feat_u])
            LABELED_INPUT_TABLES = torch.cat([LABELED_INPUT_TABLES,cur_inputs_l])
            UNLABELED_INPUT_TABLES = torch.cat([UNLABELED_INPUT_TABLES,cur_inputs_u_w])
            LABELS_TABLES = torch.cat([LABELS_TABLES,cur_labels])
            UNLABELED_LOG_TABLES = torch.cat([UNLABELED_LOG_TABLES,cur_logits_u_w])
            if len(LABELED_FEAT_TABLES) > feat_table_size_l:
                LABELED_FEAT_TABLES = LABELED_FEAT_TABLES[-feat_table_size_l:]
            if len(UNLABELED_FEAT_TABLES) > feat_table_size_u:
                UNLABELED_FEAT_TABLES = UNLABELED_FEAT_TABLES[-feat_table_size_u:]
            if len(LABELED_INPUT_TABLES) > feat_table_size_l:
                LABELED_INPUT_TABLES = LABELED_INPUT_TABLES[-feat_table_size_l:]
            if len(UNLABELED_INPUT_TABLES) > feat_table_size_u:
                UNLABELED_INPUT_TABLES = UNLABELED_INPUT_TABLES[-feat_table_size_u:]
            if len(LABELS_TABLES) > feat_table_size_l:
                LABELS_TABLES = LABELS_TABLES[-feat_table_size_l:]
            if len(UNLABELED_LOG_TABLES) > feat_table_size_u:
                UNLABELED_LOG_TABLES = UNLABELED_LOG_TABLES[-feat_table_size_u:]
        feat_l = LABELED_FEAT_TABLES
        feat_u = UNLABELED_FEAT_TABLES
        input_l = LABELED_INPUT_TABLES
        input_u = UNLABELED_INPUT_TABLES
        labels = LABELS_TABLES
        logits_u = UNLABELED_LOG_TABLES
        LABELED_FEAT_TABLES=LABELED_FEAT_TABLES.detach()
        UNLABELED_FEAT_TABLES=UNLABELED_FEAT_TABLES.detach()
        LABELED_INPUT_TABLES=LABELED_INPUT_TABLES.detach()
        UNLABELED_INPUT_TABLES=UNLABELED_INPUT_TABLES.detach()
        LABELS_TABLES=LABELS_TABLES.detach()
        UNLABELED_LOG_TABLES=UNLABELED_LOG_TABLES.detach()
    else:
        feat_l = cur_feat_l
        feat_u = cur_feat_u
        input_l = cur_inputs_l
        input_u = cur_inputs_u_w
        labels = cur_labels
        logits_u = cur_logits_u_w
    
    return feat_l, feat_u, input_l, input_u, labels, logits_u

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def infer_interleave(forward_fn, inputs_train,cfg,bs):
    merge_one_batch = cfg.merge_one_batch
    if cfg.interleave:
        inputs_interleave = list(torch.split(inputs_train, bs))
        inputs_interleave = interleave(inputs_interleave, bs)
        if merge_one_batch:
            inputs_interleave = [torch.cat(inputs_interleave)] ####
    else:
        inputs_interleave = [inputs_train]

    out_lists_inter = [forward_fn(inputs_interleave[0],return_fmap=True)]
    for inputs in inputs_interleave[1:]:
        out_lists_inter.append(forward_fn(inputs,return_fmap=True))

    for ret_id in [-1,-3,-2]:
        ret_list=[]
        for o_list in out_lists_inter:
            ret_list.append(o_list[ret_id])
        # put interleaved samples back
        if cfg.interleave:
            if merge_one_batch:
                ret_list = list(torch.split(ret_list[0], bs))
            ret_list = interleave(ret_list, bs)
            feat_l = ret_list[0]
            feat_u_w, feat_u_s = torch.cat(ret_list[1:],dim=0).chunk(2)
            #feat_l,feat_u_w, feat_u_s = ret_list
        else:
            feat_l = ret_list[0][:bs]
            feat_u_w, feat_u_s = ret_list[0][bs:].chunk(2)
        if ret_id==-1:
            logits_l,logits_u_w, logits_u_s = feat_l,feat_u_w, feat_u_s
        elif ret_id==-2:
            embedding_l, embedding_u_w, embedding_u_s = feat_l, feat_u_w, feat_u_s
        else:
            cur_feat_l = feat_l
            cur_feat_u = feat_u_w
            cur_feat_s = feat_u_s
            feat_target = torch.cat((feat_l, feat_u_w), dim=0)
    return  logits_l,logits_u_w, logits_u_s,cur_feat_l,cur_feat_u,cur_feat_s,feat_target,embedding_l, embedding_u_w, embedding_u_s

def train(epoch,train_loader , model,optimizer,lr_scheduler, cfg,device):
    if cfg.lambda_lmmd==0 and cfg.lambda_level_l==0 and cfg.lambda_level_u==0:
        loss, acc = supervised_train(epoch,train_loader, model,optimizer,lr_scheduler, cfg,device)
        return (loss, loss, 0, 0, 0, acc, 0, 0)
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    labeled_acc = AverageMeter()
    unlabeled_acc = AverageMeter()
    losses_lmmd = AverageMeter()
    losses_level_l = AverageMeter()
    losses_level_u = AverageMeter()
    masks_lmmd = AverageMeter()

    lmmd_criterion = Distribution_Loss(loss='lmmd').to(device)
    n_iter = cfg.n_imgs_per_epoch // cfg.l_batch_size
    
    end = time.time()
    for batch_idx, (data_l, data_u) in enumerate(train_loader):
        inputs_l, labels = data_l
        inputs_l, labels = inputs_l.to(device), labels.to(device)
        inputs_u_w, inputs_u_s, labels_u = data_u
        inputs_u_w, inputs_u_s, labels_u = inputs_u_w.to(device), inputs_u_s.to(device), labels_u.to(device)
        inputs_train = torch.cat((inputs_l, inputs_u_w, inputs_u_s), dim=0)
        data_time.update(time.time() - end)

        bs = inputs_l.size(0)
        cur_iteration = epoch*cfg.per_epoch_steps+batch_idx

        forward_fn = model.forward

        logits_l,logits_u_w, logits_u_s,feat_l,feat_u_w,feat_u_s,feat_target,embedding_l, embedding_u_w, embedding_u_s = infer_interleave(forward_fn, inputs_train,cfg, bs)
        labels = labels.to(torch.int64)
        L_supervised = F.cross_entropy(logits_l, labels)
        # L_supervised = loss_focal(logits_l, labels)

        L_level_l = torch.zeros_like(L_supervised)
        if cfg.lambda_level_l > 0:
            embedding_l_unripe = torch.mean(embedding_l[labels == 0], dim=0).reshape(1, 2048)
            embedding_l_ripe = torch.mean(embedding_l[labels == 1], dim=0).reshape(1, 2048)
            embedding_l_overripe = torch.mean(embedding_l[labels == 2], dim=0).reshape(1, 2048)

            cos_ripe_unripe_l = torch.cosine_similarity(embedding_l_ripe, embedding_l_unripe)
            cos_overripe_ripe_l = torch.cosine_similarity(embedding_l_overripe, embedding_l_ripe)
            cos_overripe_unripe_l = torch.cosine_similarity(embedding_l_overripe, embedding_l_unripe)

            if cos_overripe_ripe_l > cos_overripe_unripe_l and cos_ripe_unripe_l > cos_overripe_unripe_l:
                pass
            else:
                L_level_l = cos_overripe_unripe_l - cos_overripe_ripe_l
                L_level_l += cos_overripe_unripe_l - cos_ripe_unripe_l

        # unlabeled data level loss
        L_lmmd = torch.zeros_like(L_supervised)
        mmd_mask_u = torch.zeros_like(L_supervised)
        L_level_u = torch.zeros_like(L_supervised)  # unlabeled data level loss
        if cfg.lambda_lmmd>0:
            mmd_mask_l = get_mask(logits_l,cfg.lmmd_threshold,  num_class=cfg.num_classes)
            mmd_mask_u = get_mask(logits_u_w,cfg.lmmd_threshold,  num_class=cfg.num_classes)
            if mmd_mask_l.sum()>0 and mmd_mask_u.sum()>0:
                cur_feat_l, cur_feat_u, cur_input_l, cur_input_u, cur_label, cur_logits_u = \
                    update_feat_table(feat_l,feat_u_w,embedding_l,embedding_u_w,labels,logits_u_w,cfg.lmmd_feat_table_l,cfg.lmmd_feat_table_u, mask_l=mmd_mask_l, mask_u=mmd_mask_u)
                if cur_iteration>cfg.reg_warmup and len(cur_feat_l)>20 and len(cur_feat_l)==len(cur_feat_u) and cur_label.unique().shape[0] == cfg.num_classes and cur_logits_u.argmax(axis=1).unique().shape[0]==cfg.num_classes:
                    L_lmmd = lmmd_criterion(cur_feat_l, cur_feat_u, input_l=cur_input_l, input_u=cur_input_u, labels=cur_label, logits_u=cur_logits_u)

                    # unlabeled data level loss
                    if cfg.lambda_level_u > 0:
                        pseudo_label = logits_u_w.cpu().data.max(1)[1].numpy()
                        embedding_u_unripe = torch.mean(embedding_u_w[pseudo_label == 0], dim=0).reshape(1, 2048)
                        embedding_u_ripe = torch.mean(embedding_u_w[pseudo_label == 1], dim=0).reshape(1, 2048)
                        embedding_u_overripe = torch.mean(embedding_u_w[pseudo_label == 2], dim=0).reshape(1, 2048)

                        cos_ripe_unripe_u = torch.cosine_similarity(embedding_u_ripe, embedding_u_unripe)
                        cos_overripe_ripe_u = torch.cosine_similarity(embedding_u_overripe, embedding_u_ripe)
                        cos_overripe_unripe_u = torch.cosine_similarity(embedding_u_overripe, embedding_u_unripe)

                        if cos_overripe_ripe_u > cos_overripe_unripe_u and cos_ripe_unripe_u > cos_overripe_unripe_u:
                            pass
                        else:
                            L_level_u = cos_overripe_unripe_u - cos_overripe_ripe_u
                            L_level_u += cos_overripe_unripe_u - cos_ripe_unripe_u

        lambda_lmmd = scheduler.linear_warmup(cfg.lambda_lmmd, cfg.reg_warmup_iter, cur_iteration+1)
        loss = L_supervised + lambda_lmmd * L_lmmd + cfg.lambda_level_l * L_level_l + cfg.lambda_level_u * L_level_u

        # update parameters
        cur_lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # calculate accuracy for labeled data
        acc_l = (logits_l.max(1)[1] == labels).float().mean()
        acc_ul = (logits_u_w.max(1)[1] == labels_u).float().mean()

        losses.update(loss.item())
        losses_ce.update(L_supervised.item())
        losses_lmmd.update(L_lmmd.item())
        losses_level_l.update(L_level_l.item())
        losses_level_u.update(L_level_u.item())
        labeled_acc.update(acc_l.item())
        unlabeled_acc.update(acc_ul.item())
        batch_time.update(time.time() - end)
        masks_lmmd.update(mmd_mask_u.mean())
        end = time.time()

    return (losses.avg, losses_ce.avg, losses_lmmd.avg, losses_level_l.avg, losses_level_u.avg, labeled_acc.avg, unlabeled_acc.avg, masks_lmmd.avg)


def evaluate(eval_model, loader, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    # switch to evaluate mode
    eval_model.eval()
    full_targets, full_outputs=None, None
    with torch.no_grad():
        bar = Bar('Evaluating', max=len(loader))
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = eval_model(inputs)
            if full_outputs is None:
                full_targets = targets
                full_outputs = outputs
            else:
                full_targets = torch.cat((full_targets,targets))
                full_outputs = torch.cat((full_outputs,outputs))
            # measure accuracy and record loss
            targets = targets.to(torch.int64)
            loss = F.cross_entropy(outputs, targets)
            prec1 = (outputs.max(1)[1] == targets).float().mean()
            losses.update(loss.item(), inputs.shape[0])
            acc.update(prec1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if (batch_idx+1)%10==0:
                bar.suffix  = '({batch}/{size}) | Batch: {bt:.3f}s | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        bt=batch_time.avg,
                        loss=losses.avg,
                        acc=acc.avg
                        )
                bar.next()
        bar.finish()
    eval_model.train()
    return (losses.avg, acc.avg)
