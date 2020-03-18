# create by fanfan on 2020/3/13 0013
import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
Data_path = os.path.join(PROJECT_ROOT,"data")
Output_path = os.path.join(PROJECT_ROOT,'output')
Model_save_path = os.path.join(Output_path,'tf_models')



train_data_path = os.path.join(Data_path,"train.csv")
dev_data_path = os.path.join(Data_path,'test.csv')
test_data_path = os.path.join(Data_path,'test.csv')
submit_data_path = os.path.join(Output_path,'submit.csv')

label_list = ["-1","0","1"]

train_tfrecord_path = os.path.join(Output_path,'train.tf_record')
dev_tfrecord_path = os.path.join(Output_path,"dev.tf_record")
test_tfrecord_path = os.path.join(Output_path,'test.tf_record')


subjects_eng = ['系统速度','系统功能','外观','屏幕','质量','音质','性价比','服务','尺寸']
# 定义 aspect 关键词
# 这些关键词由20个二分类的 LightGBM 根据特征重要性得到
subjects = ['速度 运行 系统 网络 内存 传输速度 反应速度',
            '功能 东西 操作 体验 语音 系统 效果',
            '外观 电视 电视机 大气 东西 外形 看着',
            '幕 清晰 效果 清晰度 画质 画面 高清',
            '质量 做工 商品 质感 品质 电视 不错',
            '音效 音质 声音 屏幕 音响效果 音响 清晰',
            '性价比 便宜 值得 价格 东西 手机 实惠',
            '物流 服务 送货 师傅 服务态度 客服 态度',
            '大小 尺寸 东西 卧室 合适 不错 外观']

class ParamsModel():
    def __init__(self):
        self.max_seq_length = 256  #句子最大长度
        self.do_lower_case = True #是否区别大小写
        self.do_train = True
        self.do_eval = True
        self.do_predict = False
        self.num_train_steps = 20000
        self.warmup_step = 200
        self.save_checkpoints_steps = 200 #每训练多少步，保存一次模型
        self.max_steps_without_decrease = 20000 #最大多少步没有提升就退出
        self.learning_rate = 0.00001
        self.use_one_hot_embeddings= False
        self.optimizer = 'adamw'


        self.train_batch_size = 256
        self.buffer_size = self.train_batch_size * 30

        self.embedding_dim = 256
        self.char2id={}
        self.kernel_num = 128
        self.n_class = len(subjects_eng)
        self.n_sub_class = 3
        self.kernel_sizes = 3
        self.l2_reg = 0.0001
        self.dropout_keep = 0.5
        self.max_aspect_len = 7
        self.max_char_len = 10
        self.hiden_sizes = 200


import sys
if 'win' in sys.platform:
    bert_model_path = r'E:\nlp-data\nlp_models\albert_tiny_zh_google'
else:
    bert_model_path = r'/data/python_project/qiufengfeng/albert_zh/prev_trained_model/albert_tiny_zh_google'
bert_model_vocab_path = os.path.join(bert_model_path,'vocab.txt')
bert_model_init_path = os.path.join(bert_model_path,'albert_model.ckpt')
bert_model_config_path = os.path.join(bert_model_path,"albert_config_tiny_g.json")