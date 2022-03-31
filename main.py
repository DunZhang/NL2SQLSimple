import json

OP_DICT = {0: ">", 1: "<", 2: "==", 3: "!="}
AGG_DICT = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
CONN_DICT = {0: "", 1: "and", 2: "or"}


def find_same_col():
    """ 找sel_col和where col 有相同的 """
    with open("model_data/zhuiyi_ori_data/train/train.json", "r", encoding="utf8") as fr:
        for line in fr:
            data = json.loads(line)
            sql_info = data["sql"]
            sel_cols = sql_info["sel"]
            where_cols = [item[0] for item in sql_info["conds"]]
            if any([i in where_cols for i in sel_cols]):
                print(data)


def read_tables(file_path):
    table_data = {}
    with open(file_path, "r", encoding="utf8") as fr:
        for line in fr:
            data = json.loads(line)
            table_id = data.pop("id")
            table_data[table_id] = data
    return table_data


def normalize_query_with_value():
    table_data = read_tables("model_data/zhuiyi_ori_data/train/train.tables.json")
    with open("model_data/zhuiyi_ori_data/train/train.json", "r", encoding="utf8") as fr:
        query_data = [json.loads(line) for line in fr]
    count = 0
    for item in query_data:
        question = item["question"]
        table_id = item["table_id"]
        table = table_data[table_id]
        match_value = [j for i in table["rows"] for j in i if str(j) in question]
        if len(match_value) > 0:
            count += 1
            # print("{}\t{}".format(question, str(match_value)))
        else:
            print(question)
            # print(table)
            # print("=" * 20)
    print(len(query_data), count)


def normalize_query_with_col_name():
    table_data = read_tables("model_data/zhuiyi_ori_data/train/train.tables.json")
    with open("model_data/zhuiyi_ori_data/train/train.json", "r", encoding="utf8") as fr:
        query_data = [json.loads(line) for line in fr]
    count = 0
    for item in query_data:
        question = item["question"]
        table_id = item["table_id"]
        table = table_data[table_id]
        match_value = [i for i in table["header"] if str(i) in question]
        if len(match_value) > 0:
            count += 1
            # print("{}\t{}".format(question, str(match_value)))
        else:
            print(question)
            print(table)
            print("=" * 20)
    print(len(query_data), count)


def num_where():
    """ 调查where 子句数量 """
    counts = [0, 0, 0, 0, 0, 0, 0, 0]
    with open("model_data/zhuiyi_ori_data/train/train.json", "r", encoding="utf8") as fr:
        for line in fr:
            data = json.loads(line)
            sql_info = data["sql"]
            sel_cols = sql_info["sel"]
            where_info = sql_info["conds"]
            counts[len(where_info)] += 1
    print(counts)


if __name__ == "__main__":
    normalize_query_with_value()
