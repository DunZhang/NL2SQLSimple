from Utils.LoggerUtil import LoggerUtil
import re
from typing import List
from transformers import PreTrainedTokenizer
import torch
from os.path import join
import json

logger = LoggerUtil.get_logger()


class DataUtil():
    @staticmethod
    def read_data(data_dir_list: List[str]):
        # 读取原始数据
        sql_data, tables = [], {}
        for data_dir in data_dir_list:
            if "train" in data_dir:
                name = "train"
            elif "val" in data_dir:
                name = "val"
            elif "test" in data_dir:
                name = "test"
            else:
                name = "final"
            with open(join(data_dir, "{}.json".format(name)), "r", encoding="utf8") as fr:
                for line in fr:
                    sql_data.append(json.loads(line))
            with open(join(data_dir, "{}.tables.json".format(name)), "r", encoding="utf8") as fr:
                for line in fr:
                    data = json.loads(line)
                    table_id = data.pop("id")
                    if table_id in tables:
                        raise Exception("存在相同的table_id:{}".format(table_id))
                    tables[table_id] = data
        logger.info("question 数量：{}, table数量：{}".format(len(sql_data), len(tables)))
        return sql_data, tables

    @staticmethod
    def get_train_data(data_dir_list):
        sql_data, tables = DataUtil.read_data(data_dir_list=data_dir_list)
        # 构造为pair对
        pairs = []
        for i in sql_data:
            pairs.extend(DataUtil.gen_header_question_pair(i, tables[i["table_id"]]))
        return pairs

    @staticmethod
    def gen_header_question_pair(sql_info, table_info):
        question = sql_info["question"]
        pairs = []
        for col in table_info["header"]:
            pairs.append([str(col).strip().split(), str(question).strip().split()])
        return pairs

    @staticmethod
    def get_bert_ipt(input_ids_list: List[List[List[int]]]):
        """ 针对pair的构造 """
        input_ids, token_type_ids = [], []
        for ids1, ids2 in input_ids_list:
            input_ids.append(ids1 + ids2)
            token_type_ids.append([0] * len(ids1) + [1] * len(ids2))
        max_len = max([len(i) for i in input_ids])
        attention_mask = [[1] * len(i) + [0] * (max_len - len(i)) for i in input_ids]
        token_type_ids = [i + [0] * (max_len - len(i)) for i in input_ids]
        input_ids = [i + [0] * (max_len - len(i)) for i in input_ids]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        res = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        return res
