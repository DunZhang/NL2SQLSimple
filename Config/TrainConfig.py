from Config.AConfig import AConfig


class TrainConfig(AConfig):
    def __init__(self):
        # 训练相关
        self.num_epoch = 10
        self.batch_size = 32  #
        self.lr = 5e-5  #
        self.warmup_proportion = 0.1  #
        self.log_step = 20  #
        self.pool = "cls"  # 句向量方式
        # 模型相关
        self.pretrained_model_dir = ""
        self.model_type = "bert"  # 基本确定是roformer了
        self.device = "0"
        # 评估相关
        self.eval_step = 1000
        self.print_input_step = 200  # 多少步输出一次输入信息
        self.eval_metrics = ["acc"]
        # 数据相关
        self.train_dir = "./model_data/sim_data"
        self.dev_dir = "./model_data/sim_data"
        self.max_len = 64  # 最大长度
        # 模型保存相关
        self.output_dir = "./output/test1"
        self.seed = ""
        self.save_times_per_epoch = -1
