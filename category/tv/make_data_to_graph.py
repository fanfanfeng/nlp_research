# create by fanfan on 2017/10/11 0011
'''

python tensorflow/python/tools/free_graph.py \
--input_graph=some_graph_def.pb \ 注意：这里的pb文件是用tf.train.write_graph方法保存的
--input_checkpoint=model.ckpt.1001 \ 注意：这里若是r12以上的版本，只需给.data-00000....前面的文件名，如：model.ckpt.1001.data-00000-of-00001，只需写model.ckpt.1001  
--output_graph=/tmp/frozen_graph.pb 
--output_node_names=softmax

'''

