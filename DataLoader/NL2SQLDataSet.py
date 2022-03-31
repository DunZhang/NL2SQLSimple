import random
import torch
from torch.utils.data import Dataset
from Utils.DataUtil import DataUtil
from Utils.LoggerUtil import LoggerUtil
from transformers import PreTrainedTokenizer
from Config.TrainConfig import TrainConfig
from typing import List, Dict

logger = LoggerUtil.get_logger()


def collect_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    return DataUtil.get_bert_ipt(batch)


class NL2SQLDataSet(Dataset):
    def __init__(self, conf: TrainConfig, data_dir: str, tokenizer: PreTrainedTokenizer, data_type: str, **kwargs):
        """

        :param conf:
        :param data_path:
        :param tokenizer:
        :param data_type:
        :param kwargs:
        """
        self.conf = conf
        # 参数初始化
        self.tokenizer = tokenizer
        self.data_type = data_type
        self.data_dir = data_dir
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.init_data_model()

    def init_data_model(self):
        """ 初始化要用到的模型数据 """
        self.data = DataUtil.get_train_data([self.data_dir])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        # 获取目标数据
        col_tokens, q_tokens = self.data[item]
        col_tokens = [self.tokenizer.cls_token] + col_tokens + [self.tokenizer.sep_token]
        q_tokens = q_tokens + [self.tokenizer.sep_token]
        return self.tokenizer.convert_tokens_to_ids(col_tokens), self.tokenizer.convert_tokens_to_ids(q_tokens)
