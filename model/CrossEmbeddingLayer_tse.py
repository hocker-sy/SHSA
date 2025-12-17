import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def l2norm(X, dim, eps=1e-8):
	"""L2-normalize columns of X
    """
	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X


def maxk_pool1d_var(x, dim, k, lengths):
	"""https://github.com/woodfrog/vse_infty, thanks!"""
	results = list()
	lengths = list(lengths.cpu().numpy())
	lengths = [int(x) for x in lengths]
	for idx, length in enumerate(lengths):
		k = min(k, length)
		max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
		results.append(max_k_i)
	results = torch.stack(results, dim=0)
	return results


def maxk_pool1d(x, dim, k):
	max_k = maxk(x, dim, k)
	return max_k.mean(dim)


def maxk(x, dim, k):
	index = x.topk(k, dim=dim)[1]
	return x.gather(dim, index)


class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.output_dim = output_dim
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
		self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

	def forward(self, x):
		B, N, D = x.size()
		x = x.reshape(B * N, D)
		for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
			x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
		x = x.view(B, N, self.output_dim)
		return x


class Projector(nn.Module):
	def __init__(self, rep_dim, proj_dim):
		super().__init__()
		self.rep_dim = rep_dim
		self.proj_dim = proj_dim
		sizes = [self.rep_dim, self.rep_dim, self.proj_dim]
		layers = []
		for i in range(len(sizes) - 2):
			layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
			# layers.append(nn.BatchNorm1d(sizes[i + 1]))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
		self.projector = nn.Sequential(*layers)

	def forward(self, token_embds):
		embd_cl = self.projector(token_embds.half())
		return embd_cl.float()


class TexualEmbeddingLayer(nn.Module):
	def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
		super(TexualEmbeddingLayer, self).__init__()
		self.embed_dim = embed_dim
		self.linear = nn.Linear(input_dim, embed_dim)
		self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
		self.ratio = ratio

	def forward(self, features, text, atten):
		# print(atten) N x 77 x 77
		# features: N, 77, 512
		# text： N, 77
		mask = ((text != 0) + 0)  # N, 77
		k = int((atten.size(1) - 2) * self.ratio)
		bs = features.size(0)
		atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1  # last token
		# 是否将全局信息CLS作为K中的一个
		# atten[torch.arange(bs), :, 0] = -1  # first token
		atten = atten[torch.arange(bs), text.argmax(dim=-1), :]  # 64 x 77
		atten = atten * mask

		atten_topK = atten.topk(dim=-1, k=k)[1].unsqueeze(-1).expand(bs, k, features.size(2))  # 64 x k x 512
		features = torch.gather(input=features, dim=1, index=atten_topK)  # 64 x k x 512
		features = l2norm(features, dim=-1)

		cap_emb = self.linear(features.half())
		features = self.mlp(features) + cap_emb

		return features.float()


class VisualEmbeddingLayer(nn.Module):
	def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
		super(VisualEmbeddingLayer, self).__init__()
		self.embed_dim = embed_dim
		# self.linear = nn.Linear(input_dim, embed_dim)
		self.ratio = ratio
		self.fc = nn.Linear(input_dim, embed_dim)
		self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)

	def forward(self, base_features, atten):
		k = int((atten.size(1) - 1) * self.ratio)  # 192
		# print(k)

		# print(base_features.shape)

		bs = base_features.size(0)

		# print(bs)
		# 是否将全局信息CLS作为K中的一个
		# atten[torch.arange(bs), :, 0] = -1  # CLS token  N, 193, 193
		
		# print(atten.shape)
		
		atten_topK = atten[:, 0].topk(dim=-1, k=k)[1]  # N, k
		# print("atten_topK.shape before unsqueeze:", atten_topK.shape)
		# sys.exit()
		atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2))  # N x k x 512
		base_features = torch.gather(input=base_features, dim=1, index=atten_topK)  # N x k x 512
		base_features = l2norm(base_features, dim=-1)
		base_features = base_features.half()

		features = self.fc(base_features)  # N x k x 1024
		features = self.mlp(base_features) + features  # N x k x 1024

		return features.float()

class TexualEmbeddingLayer_adaptive(nn.Module):
	def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
		super(TexualEmbeddingLayer_adaptive, self).__init__()
		self.linear = nn.Linear(input_dim, embed_dim)
		self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
		# self.txt_weight_generator = nn.Sequential(nn.Dropout(0.1), nn.Linear(input_dim, 128),
		#                                           nn.ReLU(inplace=True), nn.Linear(128, 1))
		self.txt_weight_generator = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(inplace=True),
												  nn.Linear(input_dim, 1))
		self.ratio = ratio

	def forward(self, features, text, atten):
		# print(atten) N x 77 x 77
		# features: N, 77, 512
		# text： N, 77
		bs = features.size(0)
		mask = ((text != 0) + 0)  # N, 77
		k = int((atten.size(1) - 2) * self.ratio)
		weights = self.txt_weight_generator(features).squeeze()  # N, 77
		weights[torch.arange(bs), text.argmax(dim=-1)] = float("-inf")  # last token
		weights[torch.arange(bs), 0] = float("-inf")  # first token
		weights.masked_fill_(torch.tensor((1 - mask), dtype=torch.bool), float("-inf"))
		weights = F.softmax(weights, dim=1)

		topk_indices = torch.topk(weights, k=k, largest=True)[1]
		topk_indices = torch.sort(topk_indices, dim=1)[0]
		selected_tokens = [features[i, topk_indices[i], :] for i in range(features.shape[0])]
		selected_tokens = torch.stack(selected_tokens)  # N x k x 512
		selected_tokens = l2norm(selected_tokens, dim=-1)

		cap_emb = self.linear(selected_tokens.half())
		out_features = self.mlp(selected_tokens) + cap_emb

		return out_features.float()


class VisualEmbeddingLayer_adaptive(nn.Module):
	def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3):
		super(VisualEmbeddingLayer_adaptive, self).__init__()
		self.ratio = ratio
		self.fc = nn.Linear(input_dim, embed_dim)
		self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
		# self.image_weight_generator = nn.Sequential(nn.Dropout(0.1), nn.Linear(input_dim, 128),
		#                                             nn.ReLU(inplace=True), nn.Linear(128, 1))
		self.image_weight_generator = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(inplace=True),
												    nn.Linear(input_dim, 1))

	def forward(self, features, atten):
		k = int((atten.size(1) - 1) * self.ratio)  # 192
		bs = features.size(0)

		weights = self.image_weight_generator(features).squeeze()  # N, 77
		weights[torch.arange(bs), 0] = float("-inf")  # first token
		weights = F.softmax(weights, dim=1)

		topk_indices = torch.topk(weights, k=k, largest=True)[1]
		topk_indices = torch.sort(topk_indices, dim=1)[0]
		selected_tokens = [features[i, topk_indices[i], :] for i in range(features.shape[0])]
		selected_tokens = torch.stack(selected_tokens)  # N x k x 512
		selected_tokens = l2norm(selected_tokens, dim=-1)

		feats = self.fc(selected_tokens.half())  # N x k x 1024
		out_features = self.mlp(selected_tokens) + feats  # N x k x 1024

		return out_features.float()

def ImageTokenSelection(base_features, atten, ratio):
	k = int((atten.size(1) - 1) * ratio)  # 192
	bs = base_features.size(0)
	atten[torch.arange(bs), :, 0] = -1  # CLS token  N, 193, 193
	atten_topK = atten[:, 0].topk(dim=-1, k=k)[1]  # N, k

	atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2))  # N x k x 512
	features = torch.gather(input=base_features, dim=1, index=atten_topK)  # N x k x 512

	return features


def TextTokenSelection(features, caption_ids, atten, ratio):
	# print(atten) N x 77 x 77
	# features: N, 77, 512
	# text： N, 77
	mask = ((caption_ids != 0) + 0)  # N, 77
	lengths = mask.sum(1).view(-1) - 2  # -2 for SOS token and EOS token
	k = int((atten.size(1) - 2) * ratio)
	bs = features.size(0)
	atten[torch.arange(bs), :, caption_ids.argmax(dim=-1)] = -1  # last token
	atten[torch.arange(bs), :, 0] = -1  # first token
	atten = atten[torch.arange(bs), caption_ids.argmax(dim=-1), :]  # 64 x 77
	atten = atten * mask

	atten_topK = atten.topk(dim=-1, k=k)[1].unsqueeze(-1).expand(bs, k, features.size(2))  # N x k x 512
	features = torch.gather(input=features, dim=1, index=atten_topK)  # N x k x 512

	return features
