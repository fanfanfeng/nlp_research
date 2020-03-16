# create by fanfan on 2020/3/14 0014

import tensorflow as tf
import os
from tensorflow.contrib.predictor import from_saved_model


class EstimatorLoader():
    def __init__(self,model_path):
        self.predictor = from_saved_model(model_path)

    def get_input_keys(self):
        return self.predictor.feed_tensors.keys()

    def predict(self,dict):
        output = self.predictor(dict)
        return output

    def make_input(self,**kwargs):
        raise NotImplementedError("各种实现")




def main(model_path,tag_name=None):
    with tf.Session() as sess:
         #load the saved model
         if tag_name == None:
             tag_name = tf.saved_model.tag_constants.SERVING
         meta_graph_def = tf.saved_model.loader.load(sess, [tag_name], model_path)
         # 从meta_graph_def中取出SignatureDef对象
         signature = meta_graph_def.signature_def

         #Prepare model input, the model expects a float array to be passed to x
         # check line 28 serving_input_receiver_fn
         model_input= tf.train.Example(features=tf.train.Features(feature={
                    'x': tf.train.Feature(float_list=tf.train.FloatList(value=[6.4, 3.2, 4.5, 1.5]))
                    }))

         #get the predictor , refer tf.contrib.predicdtor
         predictor= tf.contrib.predictor.from_saved_model(model_path)

         #get the input_tensor tensor from the model graph
         # name is input_tensor defined in input_receiver function refer to tf.dnn.classifier
         input_tensor=tf.get_default_graph().get_tensor_by_name("input_tensors:0")
         #get the output dict
         # do not forget [] around model_input or else it will complain shape() for Tensor shape(?,)
         # since its of shape(?,) when we trained it
         model_input=model_input.SerializeToString()
         output_dict= predictor({"inputs":[model_input]})
         print(" prediction is " , output_dict['scores'])


if __name__ == "__main__":
    dir_path = r'E:\git-project\nlp_research\Competition\datafountain_emotion\output'
    exported_path = os.path.join(dir_path, "1584178871")
    EstimatorLoader(exported_path)