from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
import tqdm
import pdb


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
	if get_mAP:
		indices = torch.argsort(similarity, dim=1, descending=True)
	else:
		# acclerate sort with topk
		_, indices = torch.topk(
			similarity, k=max_rank, dim=1, largest=True, sorted=True
		)  # q * topk
	pred_labels = g_pids[indices.cpu()]  # q * k
	matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

	all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
	all_cmc[all_cmc > 1] = 1
	all_cmc = all_cmc.float().mean(0) * 100
	# all_cmc = all_cmc[topk - 1]

	if not get_mAP:
		return all_cmc, indices

	num_rel = matches.sum(1)  # q
	tmp_cmc = matches.cumsum(1)  # q * k

	inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
	mINP = torch.cat(inp).mean() * 100

	tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
	tmp_cmc = torch.stack(tmp_cmc, 1) * matches
	AP = tmp_cmc.sum(1) / num_rel  # q
	mAP = AP.mean() * 100

	return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
	t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10,
	                                           get_mAP=True)
	t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy(), t2i_mINP.cpu().numpy()
	if retur_indices:
		return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP,
		        t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]], indices
	else:
		return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9]]


class Evaluator:
	def __init__(self, img_loader, txt_loader):
		self.img_loader = img_loader  # gallery
		self.txt_loader = txt_loader  # query
		self.logger = logging.getLogger("FedL.eval")

	def _compute_embedding(self, model):
		model = model.eval()
		device = next(model.parameters()).device

		qids, gids, qfeats, gfeats = [], [], [], []
		# text
		for pid, caption in self.txt_loader:
			caption = caption.to(device)
			with torch.no_grad():
				text_feat = model.encode_text(caption)
			qids.append(pid.view(-1))  # flatten
			qfeats.append(text_feat)
		qids = torch.cat(qids, 0)
		qfeats = torch.cat(qfeats, 0)

		# image
		for pid, img in self.img_loader:
			img = img.to(device)
			with torch.no_grad():
				img_feat = model.encode_image(img)
			gids.append(pid.view(-1))  # flatten
			gfeats.append(img_feat)
		gids = torch.cat(gids, 0)
		gfeats = torch.cat(gfeats, 0)

		return qfeats, gfeats, qids, gids

	def _compute_embedding_tws(self, model):
		model = model.eval()
		device = next(model.parameters()).device

		qids, gids, qfeats, gfeats = [], [], [], []
		# text
		for pid, caption in self.txt_loader:
			caption = caption.to(device)
			with torch.no_grad():
				text_feat = model.encode_text_tws(caption).cpu()  # [bs, Nt, 1024]
			qids.append(pid.view(-1))  # flatten
			qfeats.append(text_feat)
		qids = torch.cat(qids, 0)
		qfeats = torch.cat(qfeats, 0)

		# image
		for pid, img in self.img_loader:
			img = img.to(device)
			with torch.no_grad():
				img_feat = model.encode_image_tws(img).cpu()  # [bs, Nv, 1024]
			gids.append(pid.view(-1))  # flatten
			gfeats.append(img_feat)
		gids = torch.cat(gids, 0)
		gfeats = torch.cat(gfeats, 0)
		# return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()
		return qfeats.cuda(), gfeats.cuda(), qids.cuda(), gids.cuda()

	def eval(self, model, i2t_metric=False):
		sims_dict = {}

		if "tws" in model.args.loss_names:
			self.logger.info("Eval with tws similarity")
			t_feats_tws, i_feats_tws, qids, gids = self._compute_embedding_tws(model)
			txt_num, txt_n_tokens, txt_dim = t_feats_tws.shape
			img_num, img_n_tokens, img_dim = i_feats_tws.shape

			i_feats_tws_norm = F.normalize(i_feats_tws, dim=-1)
			t_feats_tws_norm = F.normalize(t_feats_tws, dim=-1)
			i_feats_norm = i_feats_tws_norm.view(img_num * img_n_tokens, img_dim)  # [bs*Nv, 512]
			t_feats_norm = t_feats_tws_norm.view(txt_num * txt_n_tokens, txt_dim)  # [bs*Nt, 512]

			sim = self.batch_matmul(t_feats_norm, i_feats_norm, txt_num, txt_n_tokens, img_num, img_n_tokens)
			# sim = torch.matmul(t_feats_norm, i_feats_norm.T)  # [bs*Nt, bs*Nv]
			sim = sim.view(txt_num, txt_n_tokens, img_num, img_n_tokens)  # [bs, Nt, bs, Nv]

			similarity_tws = torch.mean(torch.max(sim, dim=-1)[0], dim=1)  # [bs, bs]
			# sim_i2t = torch.mean(torch.max(sim, dim=1)[0], dim=-1)  # [bs, bs]

			sims_dict["TWS"] = similarity_tws

		if "sdm" in model.args.loss_names or "itc" in model.args.loss_names or "tal" in model.args.loss_names:
			qfeats, gfeats, qids, gids = self._compute_embedding(model)
			qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
			gfeats = F.normalize(gfeats, p=2, dim=1)  # image features
			similarity = qfeats @ gfeats.t()
			sims_dict["Base"] = similarity
			if "tws" in model.args.loss_names:
				sims_dict["Base+TWS"] = (similarity + similarity_tws.cuda()) / 2

		table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])

		for key in sims_dict.keys():
			sims = sims_dict[key]
			if sims == None:
				continue
			rs = get_metrics(sims, qids, gids, f'{key}-t2i', False)
			table.add_row(rs)
			if i2t_metric:
				i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10,
				                                     get_mAP=True)
				i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
				table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

		table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
		table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
		table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
		table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
		table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
		table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
		table.vertical_char = " "
		self.logger.info('\n' + str(table))

		return rs[1]

	def eval_for_client(self, model):
		sims_dict = {}

		qfeats, gfeats, qids, gids = self._compute_embedding(model)
		qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
		gfeats = F.normalize(gfeats, p=2, dim=1)  # image features
		similarity = qfeats @ gfeats.t()
		sims_dict["Base"] = similarity

		# table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])

		for key in sims_dict.keys():
			sims = sims_dict[key]
			if sims == None:
				continue
			rs = get_metrics(sims, qids, gids, f'{key}-t2i', False)
		# table.add_row(rs)
		# if i2t_metric:
		# 	i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
		# 	i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
		# 	table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

		# table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
		# table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
		# table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
		# table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
		# table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
		# table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
		# table.vertical_char = " "
		# self.logger.info('\n' + str(table))

		return rs[1]

	# 分块矩阵乘：矩阵太大了，需要分块处理。
	def batch_matmul(self, t_feats, i_feats, txt_num, txt_n_tokens, img_num, img_n_tokens):
		# split, 第二个参数表示每个batch的数量
		t_feats_batch = torch.split(t_feats, int(t_feats.shape[0] / txt_n_tokens))  # 135432: 6156*22
		i_feats_batch = torch.split(i_feats, int(i_feats.shape[0] / img_n_tokens))  # 175218: 3074*57
		# pdb.set_trace()
		sim_matrix = torch.empty(txt_num * txt_n_tokens,
		                         img_num * img_n_tokens)  # (6156*txt_n_tokens, 3074*img_n_tokens)
		with torch.no_grad():
			for idx1, t_feat in tqdm.tqdm(enumerate(t_feats_batch)):
				each_row = torch.empty(txt_num, img_num * img_n_tokens)  # (6156,3074*img_n_tokens)
				for idx2, i_feat in enumerate(i_feats_batch):
					logit = torch.matmul(t_feat, i_feat.T)  # (6156,3074)
					each_row[:, img_num * idx2: img_num * (idx2 + 1)] = logit.cpu().detach()
				sim_matrix[txt_num * idx1: txt_num * (idx1 + 1), :] = each_row
		return sim_matrix
