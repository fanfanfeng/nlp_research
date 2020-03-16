# create by fanfan on 2020/3/16 0016
from utils.tf_utils.model_loader import EstimatorLoader
from utils.bert.utils import InputExample
from utils.bert import tokenization
from utils.bert.tfrecord_utils import convert_single_example
from Competition.datafountain_emotion import settings
import collections

class ModelLoader(EstimatorLoader):
    def __init__(self,model_path,label_list):
        super().__init__(model_path)
        self.label_list = label_list

    def make_input(self,text):
        params = settings.ParamsModel()
        test_label = ["0"]
        example = InputExample(guid=1,text_a=text,label="0")
        tokenizer = tokenization.FullTokenizer(
            vocab_file=settings.bert_model_vocab_path, do_lower_case=True)
        feature = convert_single_example(ex_index=0,
                                          example=example,
                                          label_list=test_label,
                                          max_seq_length=params.max_seq_length,
                                          tokenizer=tokenizer)



        features = collections.OrderedDict()
        features["input_ids"] = [feature.input_ids]
        features["input_mask"] = [feature.input_mask]
        features["segment_ids"] = [feature.segment_ids]
        features["label_ids"] = [feature.label_id]
        #features["is_real_example"] = create_int_feature(
            #[int(feature.is_real_example)])
        return features

    def predict(self,text):
        features = self.make_input("这是一个测试")
        result = self.predictor(features)
        predict_label_index = result['predictions'].tolist()[0]
        label = self.label_list[predict_label_index]
        return label




if __name__ == '__main__':
    import os
    dir_path = r'E:\git-project\nlp_research\Competition\datafountain_emotion\output'
    exported_path = os.path.join(dir_path, "1584178871")
    model_loader = ModelLoader(exported_path,settings.label_list)
    features = model_loader.make_input("这是一个测试")
    print(model_loader.predict(features))