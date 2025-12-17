import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch import nn, Tensor


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    print(f"imgae_features.shape:{image_fetures.shape}")
    print(f"text_features.shape:{text_fetures.shape}")
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    print(f"text_norm.shape: {text_norm.shape}")  
    print(f"image_norm.shape: {image_norm.shape}")  

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


# 通过加权三元组损失函数来优化图像和文本的特征匹配，使得相似的图像和文本之间的距离更小，而不相似的则距离更大
def weighted_triplet_loss(image_feat, text_feat, margin=0.6, gamma=2.0, max_violation=False):
    # scores 是图像特征和文本特征之间的相似度矩阵
    scores = image_feat @ text_feat.t()

    # 获取批大小
    bsz = image_feat.shape[0]

    # 提取对角线的分数
    diagonal = scores.diag().view(bsz, 1)

    # 扩展对角线分数，使其与scores矩阵形状相同
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # 比较每个对角线分数与其列中的分数（用于文本检索）
    cost_s = (margin + scores - d1).clamp(min=0)
    # 比较每个对角线分数与其行中的分数（用于图像检索）
    cost_im = (margin + scores - d2).clamp(min=0)

    # 创建一个掩码，用于忽略对角线上的元素
    mask = torch.eye(scores.size(0)) > .5
    if torch.cuda.is_available():
        mask = mask.cuda(device=image_feat.device)

    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    # 计算每个样本的权重
    p_s = torch.exp(-cost_s)
    weights_s = (1 - p_s) ** gamma

    p_im = torch.exp(-cost_im)
    weights_im = (1 - p_im) ** gamma

    # 加权损失
    cost_s = weights_s * cost_s
    cost_im = weights_im * cost_im

    # 如果启用 max_violation，选择最大违例（最大错误）作为损失
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    # 汇总两个方向的损失
    sum_cost_s = cost_s.sum()
    sum_cost_im = cost_im.sum()

    # 返回最终损失（取平均）
    return (sum_cost_s + sum_cost_im) / 2.0


def get_contr_loss(image_feat, text_feat, idx=None, label=None, config=None):
    """
    Args:
        image_feat, text_feat: normalized
    Returns: contrastive loss
    """
    temp = nn.Parameter(torch.ones([]) * 0.07)  # 温度参数

    # 在单GPU上直接使用图像和文本特征
    logits = image_feat @ text_feat.t() / temp  # 计算相似度矩阵并除以温度参数

    # 获取批大小
    bsz = image_feat.shape[0]

    if idx is None:
        # 如果 idx 为 None，执行标准的对比损失
        labels = torch.arange(bsz, device=image_feat.device)  # 生成标签 [0, 1, 2, ..., bsz-1]
        loss_i2t = F.cross_entropy(logits, labels)  # 图像到文本的损失
        loss_t2i = F.cross_entropy(logits.t(), labels)  # 文本到图像的损失

    else:
        # 否则，执行带有匹配矩阵的对比损失
        idx = idx.view(-1, 1)  # 将 idx 变为列向量
        assert idx.size(0) == image_feat.size(0)  # 确保 idx 和图像特征大小匹配

        # 生成匹配矩阵，判断样本是否相等
        pos_idx = torch.eq(idx, idx.t()).float()  # 生成二进制矩阵，表示哪些样本是正样本配对
        labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)  # 计算标签权重

        # 使用带权重的对比损失
        loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()  # 图像到文本损失
        loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()  # 文本到图像损失

    # 返回两个方向的损失的平均值
    return (loss_i2t + loss_t2i) / 2


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """

    return im.mm(s.t())


def euclidean_sim(x, y):
    """
      Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
      Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return 1 - dist


class func_CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(func_CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class circleLoss(nn.Module):
    """Triplet loss class

    Parameters
    ----------
    margin : float
        Ranking loss margin
    gamma : float

    metric : string
        Distance metric (either euclidean or cosine)
    """

    def __init__(self, margin=0.5, gamma=64, metric='cosine'):
        super(circleLoss, self).__init__()
        self.distance_function = euclidean_sim if metric == 'euclidean' else cosine_sim
        self.metric = metric
        self.func_circle_loss = func_CircleLoss(m=margin, gamma=gamma)
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def forward(self, im, s):
        # compute image-sentence score matrix
        # batch_size x batch_size
        scores_i2r = self.distance_function(normalize(im, dim=-1), normalize(s, dim=-1))
        scores_r2i = scores_i2r.t()
        pos = torch.eye(im.size(0))
        neg = 1 - pos

        # pos = (pos == 1).to(im.device)
        # neg = (neg == 1).to(im.device)
        pos = pos.bool().to(im.device)
        neg = neg.triu(diagonal=1).bool().to(im.device)

        scores_i2r = scores_i2r.reshape(-1)
        scores_r2i = scores_r2i.reshape(-1)
        # positive similarities
        # batch_size x 1
        sp1 = scores_i2r[pos.reshape(-1)]
        sp2 = scores_r2i[pos.reshape(-1)]

        # negative _matrix
        sn1 = scores_i2r[neg.reshape(-1)]
        sn2 = scores_r2i[neg.reshape(-1)]

        cost_im = self.func_circle_loss(sp1, sn1)
        cost_s = self.func_circle_loss(sp2, sn2)
        # clear diagonals
        ret = cost_s + cost_im
        return ret



def compute_TAL(image_features, text_features, pid, tau, margin):
    batch_size = image_features.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels  # negative

    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    alpha_i2t = ((scores / tau).exp() * labels / ((scores / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t() / tau).exp() * labels / ((scores.t() / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()

    # i2t_loss = (- (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    # t2i_loss = (- (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    # loss = torch.mean(i2t_loss) + torch.mean(t2i_loss)

    loss = (- (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0) \
           + (- (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0)
    return loss.sum()

def compute_TAL_new(sim_t2i, sim_i2t, pid, tau, margin):
    batch_size = sim_t2i.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels  # negative

    # image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    # text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    # scores = text_norm @ image_norm.t()

    alpha_i2t = ((sim_i2t / tau).exp() * labels / ((sim_i2t / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((sim_t2i.t() / tau).exp() * labels / ((sim_t2i.t() / tau).exp() * labels).sum(dim=1, keepdim=True)).detach()

    # i2t_loss = (- (alpha_i2t * scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    # t2i_loss = (- (alpha_t2i * scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    # loss = torch.mean(i2t_loss) + torch.mean(t2i_loss)

    loss = (- (alpha_i2t * sim_i2t).sum(1) + tau * ((sim_i2t / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0) \
           + (- (alpha_t2i * sim_t2i.t()).sum(1) + tau * ((sim_t2i.t() / tau).exp() * mask).sum(1).clamp(
        max=10e35).log() + margin).clamp(min=0)
    return loss.sum()


def compute_itc_new(logits_t2i, logits_i2t, pid, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    # batch_size = logits_t2i.shape[0]
    # labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    # labels = labels.to(logits_t2i.device)

    batch_size = logits_t2i.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()  # [batch_size,batch_size]
    # normalize the true matching distribution 
    labels = labels / labels.sum(dim=1)

    # labels = labels.long()
    # labels = labels.argmax(dim=1) 

    logits_per_image = logit_scale * logits_t2i
    logits_per_text = logit_scale * logits_i2t
  
    # sys.exit()
    # loss_i = F.cross_entropy(logits_per_image, labels)
    # loss_t =F.cross_entropy(logits_per_text, labels)
    loss_i = F.kl_div(F.log_softmax(logits_per_image, dim=-1), labels, reduction="batchmean")
    loss_t = F.kl_div(F.log_softmax(logits_per_text, dim=-1), labels, reduction="batchmean")
    loss = (loss_i +  loss_t)/2

    return loss


def compute_sdm_new(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    print(f"imgae_features.shape:{image_fetures.shape}")
    print(f"text_features.shape:{text_fetures.shape}")
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    # image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    # text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    txt_bs, txt_n_tokens, txt_dim = text_fetures.shape
    img_bs, img_n_tokens, img_dim = image_fetures.shape
    i_feats_norm = F.normalize(image_fetures, dim=-1)
    t_feats_norm = F.normalize(text_fetures, dim=-1)
    i_feats_norm = i_feats_norm.view(img_bs * img_n_tokens, img_dim)  # [bs*Nv, 512]
    t_feats_norm = t_feats_norm.view(txt_bs * txt_n_tokens, txt_dim)  # [bs*Nt, 512]
    
    print(f"text_norm.shape: {t_feats_norm.shape}")  
    print(f"image_norm.shape: {i_feats_norm.shape}")      
    
    t2i_cosine_theta = t_feats_norm @ i_feats_norm.t()

    # t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    print(f"i2t_pred.shape: {i2t_pred.shape}")
    print(f"image_proj_text.shape: {image_proj_text.shape}")
    print(f"labels_distribute.shape: {labels_distribute.shape}")
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss