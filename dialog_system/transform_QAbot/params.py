class hyperparams:

    # train parameter
    batch_size = 128
    eval_batch_size = 128
    lr = 0.00002
    warmup_steps = 4000
    logdir = "./output"
    num_steps = 100
    evaldir = "./model_eval"
    vocab_fpath = "./data/vocab.txt"
    data_path = 'data/xiaohuangji.csv'



    # model parameter
    vocab_size = 20000
    num_units = 512
    d_ff = 2048
    num_blocks = 6
    num_heads = 8
    maxlen = 20
    dropout_rate = 0.2
    smoothing = 0.1




    #


