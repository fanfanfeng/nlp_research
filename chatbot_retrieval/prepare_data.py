import os
import csv
import itertools
import functools
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor
from chatbot_retrieval import config
from utils.vocabprocessor import CategoricalVocabularyMy


def tokenizer_fn(iterator):
  '''
  分词方法，英文的话直接split,中文的话可以调用jieba或者自己定义分词方法
  '''
  return (x.split(" ") for x in iterator)

def create_csv_iter(filename):
  """
  读取csv文件，跳过第一行，返回一个迭代器读
  """
  with open(filename,encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for row in reader:
      yield row


def create_vocab(input_iter, min_frequency):
  """
  用tf自带的tensorflow.contrib.learn.python.learn.preprocessing类处理句子
  """
  vocab_processor = VocabularyProcessor(
      config.max_seq_len,
      min_frequency=min_frequency,
      tokenizer_fn=tokenizer_fn,
      vocabulary = CategoricalVocabularyMy(),#扩展词库，默认前面4个词是预设的
  )
  vocab_processor.fit(input_iter)
  return vocab_processor


def transform_sentence(sequence, vocab_processor):
  """
  Maps a single sentence into the integer vocabulary. Returns a python array.
  """
  return next(vocab_processor.transform([sequence])).tolist()


def create_text_sequence_feature(fl, sentence, sentence_len, vocab):
  """
  Writes a sentence to FeatureList protocol buffer
  """
  sentence_transformed = transform_sentence(sentence, vocab)
  for word_id in sentence_transformed:
    fl.feature.add().int64_list.value.extend([word_id])
  return fl


def create_example_train(row, vocab):
  """
  Creates a training example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, utterance, label = row
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  label = int(float(label))

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example


def create_example_test(row, vocab):
  """
  Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, utterance = row[:2]
  distractors = row[2:]
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])

  # Distractor sequences
  for i, distractor in enumerate(distractors):
    dis_key = "distractor_{}".format(i)
    dis_len_key = "distractor_{}_len".format(i)
    # Distractor Length Feature
    dis_len = len(next(vocab._tokenizer([distractor])))
    example.features.feature[dis_len_key].int64_list.value.extend([dis_len])
    # Distractor Text Feature
    dis_transformed = transform_sentence(distractor, vocab)
    example.features.feature[dis_key].int64_list.value.extend(dis_transformed)
  return example


def create_tfrecords_file(input_filename, output_filename, example_fn):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for i, row in enumerate(create_csv_iter(input_filename)):
    x = example_fn(row)
    writer.write(x.SerializeToString())
  writer.close()
  print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
  """
  Writes the vocabulary to a file, one word per line.
  """
  vocab_size = len(vocab_processor.vocabulary_)
  with open(outfile, "w",encoding='utf-8') as vocabfile:
    for id in range(vocab_size):
      word =  vocab_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


if __name__ == "__main__":
  if not os.path.exists(config.vocabulary_path):
    print("创建词库...")
    input_iter = create_csv_iter(config.TRAIN_PATH)
    input_iter = (x[0] + " " + x[1] for x in input_iter)
    vocab = create_vocab(input_iter, min_frequency=config.min_word_frequency)
    print("词库大小: {}".format(len(vocab.vocabulary_)))
    # Create vocabulary.txt file
    write_vocabulary(vocab, config.vocabulary_path)
    # 保存词汇库，后面直接restore
    vocab.save(config.vocabulary_path_bin)
  else:
    vocab = VocabularyProcessor.restore(config.vocabulary_path_bin)


  # Create validation.tfrecords
  create_tfrecords_file(
      input_filename=VALIDATION_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "validation.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create test.tfrecords
  create_tfrecords_file(
      input_filename=TEST_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "test.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create train.tfrecords
  create_tfrecords_file(
      input_filename=TRAIN_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
      example_fn=functools.partial(create_example_train, vocab=vocab))
