# create by fanfan on 2020/1/8 0008
import os
import pandas as pd
import tensorflow as tf
from tensorflow_estimator.contrib.estimator import DNNClassifierWithLayerAnnotations
import  tensorflow_estimator as tf_estimator

FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
dir_path = r'E:\nlp-data\train_data\iris'
train_path = os.path.join(dir_path,'iris_training.csv')
test_path = os.path.join(dir_path,'iris_test.csv')
train = pd.read_csv(train_path,names=FUTURES,header=0)
train_x,train_y = train,train.pop("Species")

test = pd.read_csv(test_path,names=FUTURES,header=0)
test_x,test_y = test,test.pop('Species')

feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))
print(test_y)


def my_model_fn(features,labels,mode,params):
    net = tf.feature_column.input_layer(features,params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.layers.dense(net,units=units,activation=tf.nn.relu)

    logits = tf.layers.dense(net,params['n_classes'],activation=None)

    predicted_classes = tf.argmax(logits,1)
    if mode == tf_estimator.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids':predicted_classes[:,tf.newaxis],
            'probabilities':tf.nn.softmax(logits),
            'logits':logits
        }
        return tf_estimator.estimator.EstimatorSpec(mode,predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    if mode == tf_estimator.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf_estimator.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    precision = tf.metrics.precision(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    recall = tf.metrics.recall(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    mertric = {'accuracy':accuracy,'precision':precision,'recall':recall}
    tf.summary.scalar('accuracy',accuracy[1])
    if mode == tf_estimator.estimator.ModeKeys.EVAL:
        return tf_estimator.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=mertric)



models_path=os.path.join(dir_path,'mymodels/')
classifier = tf_estimator.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=models_path,
    params={
        'feature_columns':feature_columns,
        'hidden_units':[10,10],
        'n_classes':3})

def train_input_fn(features,labels,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

tf.logging.set_verbosity(tf.logging.INFO)
batch_size = 100
classifier.train(input_fn=lambda :train_input_fn(train_x,train_y,batch_size),steps=1000)


def eval_input_fn(features,labels,batch_size):
    features = dict(features)
    inputs = (features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

eval_result = classifier.evaluate(
    input_fn=lambda :eval_input_fn(test_x,test_y,batch_size)
)

print(eval_result)