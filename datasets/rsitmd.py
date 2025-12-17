# -*- coding: utf-8 -*-
# @Time       : 2024/9/29 17:06
# @Author     : Marverlises
# @File       : rsitmd.py
# @Description: PyCharm
import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class RSITMD(BaseDataset):
    """
    RSICD

    Reference:
    Exploring Models and Data for Remote Sensing Image Caption Generation (IEEE TGRS 2017)


    Dataset statistics:
    ### classes: 30
    ### images: 10921,  (train)  (test)  (val)
    ### captions:
    ###  724 images are described by five different sentences
    1495 images are described by four different sentences
    2182 images are described by three different sentences
    1667 images are described by two different sentences
    4853 images are described by one sentence

    annotation format:
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'RSITMD'

    # 在类中定义__init__()方法，方便创建实例的时候，需要给实例绑定上属性，也方便类中的方法（函数）的定义。
    # root为路径
    def __init__(self, root='', verbose=True):
        super(RSITMD, self).__init__()
        # root = r'/root/autodl-tmp/data/IRRA'
        # super().__init__()就是继承父类的init方法，同样可以使用super()去继承其他方法。
        self.dataset_dir = op.join(root, self.dataset_dir)
        # self.img_dir = op.join(self.dataset_dir, 'images/')
        self.img_dir = ''
        
        self.anno_path = op.join(self.dataset_dir, 'Transferdataset_rsitmd.json')
        # op.join直接连接内部的路径，即最后输出的结果为root/self.dataset_dir、self.dataset_dir/imgs/
        self._check_before_run()
        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RSITMD Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    # 处理注释数据，并根据是否是训练模式返回不同格式的数据集和PID容器
    def _process_anno(self, annos: List[dict], training=False):
        imgid_container = set() #  存储唯一的行人ID
        dataset = []
        image_id = 0  # img计数器
        if training:
            for anno in annos:
                imgid = image_id # make imgid begin from 0
                imgid_container.add(imgid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions']  # caption list
                for caption in captions:
                    # 将包含PID,imgID,imgPath和caption的元组加入到dataset中
                    dataset.append((imgid, image_id, img_path, caption))
                image_id += 1
            return dataset, imgid_container
        else:
            # 非training
            img_paths = []
            captions = []
            image_imgids = []
            caption_imgids = []
            for anno in annos:
                imgid = int(anno['id'])
                imgid_container.add(imgid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_imgids.append(imgid)
                caption_list = anno['captions']  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_imgids.append(imgid)
            dataset = {
                "image_pids": image_imgids,
                "img_paths": img_paths,
                "caption_pids": caption_imgids,
                "captions": captions
            }
            return dataset, imgid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
