import sys
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from .objectives import circleLoss
from .CrossEmbeddingLayer_tse import VisualEmbeddingLayer, TexualEmbeddingLayer


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)

        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5
            # 创建三个层归一化层，分别用于输入的文本、图像和最终输出
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        # 初始化Circle Loss
        if args.circle_loss:
            # 初始化Circle Loss各参数
            self.circle_loss = circleLoss(args.circle_margin, args.circle_gamma, args.circle_metric)
            #是否要使用circle_loss
            self.circle_task = True

            # token-wise interaction,先对tokens坐一次token selection+embedding，以减少计算量
        if "tws" in args.loss_names:
            self.tws_visual = VisualEmbeddingLayer(input_dim=512, embed_dim=1024, ratio=0.3)
            self.tws_text = TexualEmbeddingLayer(input_dim=512, embed_dim=1024, ratio=0.3)
        

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
        # self.logger.info(f'Training Model with {self.current_task} tasks')

    # 定义跨模态变换函数，接收查询（q）、键（k）和值（v）
    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        # x, weight = self.base_model.visual.attnpool(image) 
        x, weight = self.base_model.encode_image(image) 
        # x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x, weight = self.base_model.encode_text(text)
        # x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()
    
     #tws相关图片嵌入   
    def encode_image_tws(self, image):
        x, weight = self.base_model.encode_image(image)
        i_feat_tws = self.tws_visual(x, weight)
        return i_feat_tws.float()
    #tws相关文本嵌入 
    def encode_text_tws(self, text):
        x, weight = self.base_model.encode_text(text.long())
        t_feat_tws = self.tws_text(x, text, weight)
        return t_feat_tws.float()


    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        # image_feats, text_feats = self.base_model(images, caption_ids)
        # i_feats = image_feats[:, 0, :].float()
 
        image_feats, atten_img, text_feats, atten_txt = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        # i_feats = image_feats.float() # for CLIP ResNet visual model
        # t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if self.circle_task:
            cir_loss = self.circle_loss(i_feats, t_feats)
            ret.update({'circle_loss': cir_loss})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'wtriplet' in self.current_task:
            ret.update({'wtriplet_loss': objectives.weighted_triplet_loss(i_feats, t_feats)})
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'circle' in self.current_task:
            ret.update({'circle_loss': circleLoss(i_feats, t_feats)})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})


        # token-wise interaction
        if "tws" in self.current_task:
            # tws_visual() 和 tws_text() 负责 从图像和文本中提取 token 级别的特征
            i_feats_ts = self.tws_visual(image_feats, atten_img)  # [bs, Nv, 1024]
            t_feats_ts = self.tws_text(text_feats, caption_ids, atten_txt)  # [bs, Nt, 1024]

            txt_bs, txt_n_tokens, txt_dim = t_feats_ts.shape
            img_bs, img_n_tokens, img_dim = i_feats_ts.shape
            # 归一化特征向量
            i_feats_norm = F.normalize(i_feats_ts, dim=-1)
            t_feats_norm = F.normalize(t_feats_ts, dim=-1)
            i_feats_norm = i_feats_norm.view(img_bs * img_n_tokens, img_dim)  # [bs*Nv, 512]
            t_feats_norm = t_feats_norm.view(txt_bs * txt_n_tokens, txt_dim)  # [bs*Nt, 512]
            # 计算文本token和图像token之间的余弦相似度
            sim = torch.matmul(t_feats_norm, i_feats_norm.T)  # [bs*Nt, bs*Nv]
            sim = sim.view(txt_bs, txt_n_tokens, img_bs, img_n_tokens)  # [bs, Nt, bs, Nv]
            sim_t2i = torch.mean(torch.max(sim, dim=-1)[0], dim=1)  # [bs, bs]
            sim_i2t = torch.mean(torch.max(sim, dim=1)[0], dim=-1)  # [bs, bs]
            # print(sim_t2i.dtype)
            # print(sim_i2t.dtype)
            # sys.exit()
            # sim_t2i = sim_t2i.long()
            # sim_i2t = sim_i2t.long()
            # print(batch['pids'].dtype)
            # sys.exit()
            tws_loss = objectives.compute_itc_new(sim_t2i, sim_i2t, batch['pids'], logit_scale)
            # tws_loss = objectives.compute_sdm_new(i_feats_ts, t_feats_ts, batch['pids'], logit_scale)
            # tws_loss = objectives.compute_TAL_new(sim_t2i, sim_i2t, batch['pids'], self.args.tau, self.args.margin)
            # loss = objectives.compute_sdm_new(sim_t2i, sim_i2t, batch['pids'], logit_scale)
            ret.update({'tws_loss': tws_loss})

        if 'tal' in self.current_task:
            ret.update(
                {'tal_loss': objectives.compute_TAL(i_feats, t_feats, batch['pids'], self.args.tau, self.args.margin)})

        if "ritc" in self.current_task:  # Reversed Image-Text Contrastive Loss
            pid = batch['pids'].reshape((i_feats.shape[0], 1))  # make sure pid size is [batch_size, 1]
            pid_dist = pid - pid.t()
            labels = (pid_dist == 0).float()

            image_features_norm = F.normalize(i_feats)
            text_features_norm = F.normalize(t_feats)
            logits_per_image = logit_scale * image_features_norm @ text_features_norm.t()
            # logits_per_text = logit_scale * text_features_norm @ image_features_norm.t()
            logits_per_text = logits_per_image.t()

            img_log = F.log_softmax(logits_per_image, dim=1)  # pi,j
            txt_log = F.log_softmax(logits_per_text, dim=1)  # pj,i
            target_log = (labels + 1e-2).log()  # log(qi,j + ε)
            kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
            kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
            ritc_loss = 0.5 * (kl_img + kl_txt)
            ret['ritc_loss'] = ritc_loss  # 1.0
            
        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
