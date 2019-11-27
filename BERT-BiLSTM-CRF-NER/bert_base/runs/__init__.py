# create by fanfan on 2019/2/25 0025

def start_server():
    from bert_base.server import BertServer
    from bert_base.server.helper import get_run_args

    args = get_run_args()
    server = BertServer(args)
    server.start()
    server.join()

def start_client():
    pass

def train_ner():
    import os
    from bert_base.train.train_helper import get_args_parser
    from bert_base.train.bert_lstm_ner import train
    args = get_args_parser()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k,v) for k,v in sorted(vars(args).items())])
        print('usage: %s\n%20s    %s\n%s\n%s\n' % (" ".join(sys.argv),'ARG','VALUE','_'*50,param_str))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)


if __name__ == '__main__':
    start_server()