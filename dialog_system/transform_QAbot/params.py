import os
probject_root = os.path.dirname(os.path.abspath(__file__))
class Params:
    # train parameter
    batch_size = 256
    eval_batch_size = 256
    lr = 0.00002
    warmup_steps = 500
    logdir = os.path.join(probject_root,"output")
    output_path = os.path.join(probject_root,"output")
    num_steps = 100
    evaldir = os.path.join(output_path,"model_eval")
    evaluate_every_steps = 20
    vocab_fpath = os.path.join(output_path,"vocab.txt")
    origin_data = os.path.join(probject_root,'data')
    device_map = '0'


    # model parameter
    vocab_size = 40000
    num_units = 512
    d_ff = 2048
    num_blocks = 6
    num_heads = 8
    max_seq_length = 20
    dropout_rate = 0.2
    smoothing = 0.1
    total_steps = 60000
    filter_size =3




class TestParams:
    # train parameter
    batch_size = 10
    eval_batch_size = 10
    lr = 0.00002
    warmup_steps = 500
    logdir = os.path.join(probject_root, "output")
    output_path = os.path.join(probject_root, "output")
    num_steps = 100
    evaldir = os.path.join(output_path, "model_eval")
    evaluate_every_steps = 10
    vocab_fpath = os.path.join(output_path, "vocab.txt")
    origin_data = os.path.join(probject_root, 'data')
    device_map = '0'

    # model parameter
    vocab_size = 100
    num_units = 128
    d_ff = 2048
    num_blocks = 2
    num_heads = 2
    max_seq_length = 20
    dropout_rate = 0.2
    smoothing = 0.1
    total_steps = 600
    filter_size =3


