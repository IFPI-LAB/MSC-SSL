from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def compute_kernel(x, y):
	x_size = x.size(0)
	y_size = y.size(0)
	dim = x.size(1)
	x = x.unsqueeze(1)  # (x_size, 1, dim)
	y = y.unsqueeze(0)  # (1, y_size, dim)
	tiled_x = x.expand(x_size, y_size, dim)
	tiled_y = y.expand(x_size, y_size, dim)
	kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
	return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd_loss(x, y,reduction=None):
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
	return mmd


def reg_MMD(fm_labeled, fm_unlabeled):
	fml = F.adaptive_avg_pool2d(fm_labeled, (1, 1))
	fmu = F.adaptive_avg_pool2d(fm_unlabeled, (1, 1))
	fml = fml.reshape((fml.shape[0], -1))
	fmu = fmu.reshape((fmu.shape[0], -1))
	fea_loss = mmd_loss(fml, fmu)
	# print(torch.norm(fml-fmu), fea_loss)
	return fea_loss


def reg_KD(fm_src, fm_tgt,):
	b, c, h, w = fm_src.shape
	fea_loss = torch.norm(fm_src - fm_tgt) / (h * c )
	return fea_loss


def softmax_mse_loss(input_logits, target_logits,reduction='mean'):
	"""Takes softmax on both sides and returns MSE loss

	Note:
	- Returns the sum over all examples. Divide by the batch size afterwards
	  if you want the mean.
	- Sends gradients to inputs but not the targets.
	"""
	input_softmax = F.softmax(input_logits, dim=1)
	target_softmax = F.softmax(target_logits, dim=1)
	#num_classes = input_logits.size()[1]
	return F.mse_loss(input_softmax, target_softmax, reduction=reduction) #/ num_classes


def softmax_kl_loss(input_logits, target_logits,reduction='mean'):
	"""Takes softmax on both sides and returns KL divergence

	Note:
	- Returns the sum over all examples. Divide by the batch size afterwards
	  if you want the mean.
	- Sends gradients to inputs but not the targets.
	"""
	input_log_softmax = F.log_softmax(input_logits, dim=1)
	target_softmax = F.softmax(target_logits, dim=1)
	return F.kl_div(input_log_softmax, target_softmax, reduction=reduction)


def softmax_lmmd_loss(source, target, source_label, target_logits):
	target_softmax = F.softmax(target_logits, dim=1)
	batch_size = source.size()[0]
	weight_ss, weight_tt, weight_st = cal_weight(source_label, target_softmax)
	weight_ss = torch.from_numpy(weight_ss).cuda()  # B, B
	weight_tt = torch.from_numpy(weight_tt).cuda()
	weight_st = torch.from_numpy(weight_st).cuda()

	kernels = guassian_kernel(source, target)
	loss = torch.Tensor([0]).cuda()
	if torch.sum(torch.isnan(sum(kernels))):
		return loss
	SS = kernels[:batch_size, :batch_size]
	TT = kernels[batch_size:, batch_size:]
	ST = kernels[:batch_size, batch_size:]

	# debug
	weight_ss_np = weight_ss.cpu().detach().numpy()
	weight_tt_np = weight_tt.cpu().detach().numpy()
	weight_st_np = weight_st.cpu().detach().numpy()

	SS_np = SS.cpu().detach().numpy()
	TT_np = TT.cpu().detach().numpy()
	ST_np = ST.cpu().detach().numpy()

	ss_mul = weight_ss_np * SS_np
	tt_mul = weight_tt_np * TT_np
	st_mul = weight_st_np * ST_np

	loss_np_ = ss_mul + tt_mul - 2 * st_mul
	loss_np = np.sum(ss_mul + tt_mul - 2 * st_mul)

	loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
	# Dynamic weighting
	# lamb = self.lamb()
	# self.step()
	# loss = loss * lamb
	return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	n_samples = int(source.size()[0]) + int(target.size()[0])
	total = torch.cat([source, target], dim=0)
	total0 = total.unsqueeze(0).expand(
		int(total.size(0)), int(total.size(0)), int(total.size(1)))
	total1 = total.unsqueeze(1).expand(
		int(total.size(0)), int(total.size(0)), int(total.size(1)))
	L2_distance = ((total0 - total1) ** 2).sum(2)
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
	bandwidth /= kernel_mul ** (kernel_num // 2)
	bandwidth_list = [bandwidth * (kernel_mul ** i)
					  for i in range(kernel_num)]
	kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
				  for bandwidth_temp in bandwidth_list]
	return sum(kernel_val)


def cal_weight(source_label, target_logits):
	num_class = target_logits.size()[1]
	batch_size = source_label.size()[0]
	source_label = source_label.cpu().data.numpy()
	source_label_onehot = np.eye(num_class)[source_label]  # one hot

	source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, num_class)  # 1*num_class维，每类样本的数量
	source_label_sum[source_label_sum == 0] = 100
	source_label_onehot = source_label_onehot / source_label_sum  # label ratio

	# Pseudo label
	target_label = target_logits.cpu().data.max(1)[1].numpy()

	target_logits = target_logits.cpu().data.numpy()
	target_logits_sum = np.sum(target_logits, axis=0).reshape(1, num_class)
	target_logits_sum[target_logits_sum == 0] = 100
	target_logits = target_logits / target_logits_sum

	weight_ss = np.zeros((batch_size, batch_size))
	weight_tt = np.zeros((batch_size, batch_size))
	weight_st = np.zeros((batch_size, batch_size))

	set_s = set(source_label)
	set_t = set(target_label)
	count = 0
	for i in range(num_class):  # (B, C)
		if i in set_s and i in set_t:
			s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
			t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

			ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
			weight_ss = weight_ss + ss
			tt = np.dot(t_tvec, t_tvec.T)
			weight_tt = weight_tt + tt
			st = np.dot(s_tvec, t_tvec.T)
			weight_st = weight_st + st
			count += 1

	length = count
	if length != 0:
		weight_ss = weight_ss / length
		weight_tt = weight_tt / length
		weight_st = weight_st / length
	else:
		weight_ss = np.array([0])
		weight_tt = np.array([0])
		weight_st = np.array([0])
	return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


class Distribution_Loss(nn.Module):
	def __init__(self, loss='softmax_mse', reduction='mean'):#['mse','kl','softmax_mse','softmax_kl']
		super(Distribution_Loss, self).__init__()
		self.check_shape = True
		loss=loss.lower()
		if loss=='mse':
			criterion = F.mse_loss
		elif loss=='softmax_mse':
			criterion = softmax_mse_loss
		elif loss=='kl':
			criterion = F.kl_div
		elif loss=='softmax_kl':
			criterion = softmax_kl_loss
		elif loss=='mmd':
			criterion = mmd_loss
			self.check_shape = False
		elif loss=='lmmd':
			criterion = softmax_lmmd_loss
			self.check_shape = False
		else:
			raise NotImplementedError
		self.loss_name = loss
		self.criterion = criterion
		self.reduction = reduction

	def forward(self, input_logits, target_logits, mask=None, reduction=None, input_l=None, input_u=None, labels=None, logits_u=None):
		if self.check_shape:
			assert input_logits.size() == target_logits.size()
		if reduction is None:
			reduction=self.reduction

		input_logits = F.adaptive_avg_pool2d(input_logits, (1, 1))
		target_logits = F.adaptive_avg_pool2d(target_logits, (1, 1))
		input_logits = input_logits.reshape((input_logits.shape[0], -1))
		target_logits = target_logits.reshape((target_logits.shape[0], -1))
		if self.loss_name == 'lmmd':
			loss = self.criterion(input_l, input_u, labels, logits_u)
		else:
			loss = self.criterion(input_logits, target_logits,reduction=reduction)
		if 'softmax' not in self.loss_name and 'mmd' not in self.loss_name:
			loss = loss/10000
		if len(loss.shape)>1:
			loss=loss.sum(1)
			if mask is not None:
				loss = (loss*mask).sum()/(mask.sum() if mask.sum()>0 else 1)
			else:
				loss = loss.mean()
		#loss=loss/2.5
		return loss
