# create by fanfan on 2020/3/13 0013
import sys
sys.path.append(r"/data/python_project/nlp_research")
from utils.bert.processor import DataProcessor
from Competition.ai_challenger_2018_sentiment_analysis import settings
from utils.bert import tokenization
from utils.bert.utils import InputExample
from utils.bert.tfrecord_utils import convert_single_example
import os
import tensorflow as tf
import pandas as pd
import collections
import numpy as np

model_params = settings.ParamsModel()

class AiSentimentProcess(DataProcessor):
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            print(i)
            text_a = tokenization.convert_to_unicode(line[-1])
            label = self.change_label_to_id(line[:-1])
            examples.append(
                InputExample(guid=i, text_a=text_a, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            text_a = tokenization.convert_to_unicode(line[-1])
            label = self.change_label_to_id(line[:-1])
            examples.append(
                InputExample(guid=i, text_a=text_a, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            text_a = tokenization.convert_to_unicode(line[-1])
            label = self.change_label_to_id(line[:-1])
            examples.append(
                InputExample(guid=i, text_a=text_a,label=label))
        return examples

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return settings.label_list

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = pd.read_csv(input_file)
        for items in data.values:
            yield items

    def change_label_to_id(self,labels):
        label_ids = []
        for value  in labels:
            tmp = [0] * model_params.n_sub_class
            tmp[settings.label_list.index(str(value))] = 1
            label_ids += tmp
        return label_ids



def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        bak = example.label
        example.label = "0"
        feature = convert_single_example(ex_index, example, label_list,
                             max_seq_length, tokenizer)
        feature.label_id = bak

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def create_tfrecorf_file():
    processor = AiSentimentProcess()
    label_list = processor.get_labels()
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


def file_based_input_fn_builder(input_file, seq_length, is_training,n_label,n_sub_class
                                ,batch_size,buffer_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([n_label*n_sub_class], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            if name == 'label_ids':
                t = tf.reshape(t,[n_label,n_sub_class])
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=buffer_size)

        d = d.apply(
            tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size)
        )

        return d

    return input_fn

def make_aspect_array():
    tokenizer = tokenization.FullTokenizer(
        vocab_file=settings.bert_model_vocab_path, do_lower_case=True)
    total_sentence_char_ids = []
    for  subject in settings.subjects:
        sub_list = []
        for index, i in enumerate(subject.split(" ")):
            example  = InputExample(guid=index,text_a=i,label="0")
            feature = convert_single_example(index,example,settings.label_list,max_seq_length=10,tokenizer=tokenizer)
            sub_list.append(feature.input_ids)
        total_sentence_char_ids.append(sub_list)
    return np.array(total_sentence_char_ids,dtype=np.int64)

if __name__ == '__main__':
    create_tfrecorf_file()

