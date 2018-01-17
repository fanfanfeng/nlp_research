# create by fanfan on 2017/9/19 0019
import numpy as np



def cret_dict():
    with open('data/vocab40000.in',encoding='utf-8') as f:
        set_words = [line.strip() for line in f ]
        int_to_vab = { word_i: word for word_i, word in enumerate(set_words)}
        vab_to_int = { word: word_i for word_i, word in int_to_vab.items()}
        return int_to_vab, vab_to_int


# =======================
int_to_source, source_to_int = cret_dict()
int_to_target, target_to_int = int_to_source, source_to_int
# 对字母进行转化


import tensorflow as tf
from tensorflow.python.layers.core import Dense


def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                      sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['_GO']), ending], 1)

    return decoder_input


def decoding_layer(target_to_int, decoding_embedding_size, num_layers, rnn_size, target_sequence_length,
                   max_target_sequence_length, encoder_state, decoder_input):
    target_vocab_size = len(target_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    #
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)
        training_decoder_output, _,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)

    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_to_int['_GO']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     target_to_int['_EOS'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _ ,_= tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
    #
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    decoder_input = process_decoder_input(targets, target_to_int, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(target_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)

    return training_decoder_output, predicting_decoder_output


epochs = 60

batch_size = 128

rnn_size = 128

num_layers = 2

encoding_embedding_size = 128
decoding_embedding_size = 128

learning_rate = 0.001

train_graph = tf.Graph()

with train_graph.as_default():
    # 获得模型输入
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_to_int),
                                                                       len(target_to_int),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)

    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


import data_utils
train_batch_manager = data_utils.BatchManager('data/train.txt.id40000.in',128)
test_batch_manager = data_utils.BatchManager("data/test.txt.id40000.in",128)

display_step = 50  # 每隔50轮输出loss

# ============================
current_epoch = 0

best_validation_loss = 20
checkpoint = "trained_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs + 1):

        for batch_i, train_batch in enumerate(
                train_batch_manager.iterbatch()):
            _, loss = sess.run(
                [train_op, cost],
                {input_data: train_batch['encode'],
                 targets: train_batch['decode'],
                 lr: learning_rate,
                 target_sequence_length: train_batch['decode_lengths'],
                 source_sequence_length: train_batch['encode_lengths']})

            if batch_i % display_step == 0:
                # 计算validation loss
                for i, test_batch in enumerate(train_batch_manager.iterbatch()):
                    validation_loss = sess.run(
                        [cost],
                        {input_data: test_batch['encode'],
                         targets: test_batch['decode'],
                         lr: learning_rate,
                         target_sequence_length: test_batch['decode_lengths'],
                         source_sequence_length: test_batch['encode_lengths']})

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  12800 // batch_size,
                                  loss,
                                  validation_loss[0]))

    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Saved')