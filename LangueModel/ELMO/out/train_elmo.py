# create by fanfan on 2018/11/23 0023


from LangueModel.ELMO.bilm.train import train,load_vocab
from LangueModel.ELMO.bilm.data import BidirectionalLMDataSet
import argparse

def main(args):
    max_word_len = 20
    vocab = load_vocab(args.vocab_file,max_word_len)

    # define the options
    batch_size = 58
    n_gpus = 1
    n_train_tokens = 1522

    options = {
        'bidirectional':True,
        'dropout':0.1,
        'lstm':{
            'cell_clip':3,
            'dim':512,
            'n_layers':2,
            'proj_clip':3,
            'projection_dim':100,
            'use_skip_connections':True
        },
        'all_clip_norm_val':10.0,
        'n_epochs':10,
        'n_train_tokens':n_train_tokens,
        'batch_size':batch_size,
        'n_tokens_vocab':vocab.size,
        'unroll_steps':20,
        'n_negative_samples_batch':500
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataSet(prefix,vocab,test=False,shuffle_on_load=True)

    tf_save_dir  = args.save_dir
    tf_log_dir = args.save_dir

    train(options,data,n_gpus,tf_save_dir,tf_log_dir)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--save_dir',default='log/',help="Location of checkpoint files")
    parse.add_argument('--vocab_file',default='result/vocab.txt',help='Vocabulary file')
    parse.add_argument('--train_prefix',default='data/*',help='Prefix for train files')

    args = parse.parse_args()
    main(args)
