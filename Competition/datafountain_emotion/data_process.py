# create by fanfan on 2020/3/13 0013
from utils.bert.processor import DataProcessor
from Competition.datafountain_emotion import settings
from utils.bert import tokenization
from utils.bert.utils import InputExample
from utils.bert.tfrecord_utils import file_based_convert_examples_to_features
import os



class DataFountainEmotionProcess(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a,label="0"))
        return examples

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return settings.label_list


def create_tfrecorf_file():
    processor = DataFountainEmotionProcess()
    label_list = processor.get_labels()
    model_params = settings.ParamsModel()
    # 分词器，不支持中文分词，
    tokenizer = tokenization.FullTokenizer(
        vocab_file=settings.bert_model_vocab_path, do_lower_case=True)
    if not os.path.exists(settings.train_tfrecord_path):
        train_examples = processor.get_train_examples(settings.train_data_path)
        file_based_convert_examples_to_features(
            train_examples, label_list, model_params.max_seq_length, tokenizer, settings.train_tfrecord_path)

    if not os.path.exists(settings.dev_tfrecord_path):
        dev_examples = processor.get_dev_examples(settings.dev_data_path)
        file_based_convert_examples_to_features(
            dev_examples, label_list, model_params.max_seq_length, tokenizer, settings.dev_tfrecord_path)

    if not os.path.exists(settings.test_tfrecord_path):
        test_examples = processor.get_test_examples(settings.test_data_path)
        file_based_convert_examples_to_features(
            test_examples, label_list, model_params.max_seq_length, tokenizer, settings.test_tfrecord_path)

if __name__ == '__main__':
    create_tfrecorf_file()
    test_string = "##武汉加油##"
    print(tokenization.convert_to_unicode(test_string))
    tokenizer = tokenization.FullTokenizer(
        vocab_file=settings.bert_model_vocab_path, do_lower_case=True)
    print(tokenizer.tokenize(test_string))

