# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
wechat: lp9628
blog: http://blog.csdn.net/u014365862/article/details/78422372
"""

# inception_v4:299 
# resnet_v2:224
# vgg:224

IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299
num_classes = 4
EARLY_STOP_PATIENCE = 1000
# epoch
epoch = 1000
batch_size = 1
# 模型的学习率
learning_rate = 0.00001
keep_prob = 0.8


# 设置训练样本的占总样本的比例：
train_rate = 0.9

# 每个类别保存到一个文件中，放在此目录下，只要是二级目录就可以。
craterDir = "sample_train"

# 选择需要的模型
# arch_model="arch_inception_v4";
# arch_model="arch_resnet_v2_50"
# arch_model="vgg_16"
arch_model="arch_inception_v4_rnn_attention"

# 设置要更新的参数和加载的参数，目前是非此即彼，可以自己修改哦
checkpoint_exclude_scopes = "Logits_out"

# 迁移学习模型参数， 下载训练好模型：https://github.com/MachineLP/models/tree/master/research/slim
# checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"; 
# checkpoint_path="pretrain/resnet_v2_50/resnet_v2_50.ckpt"
checkpoint_path="pretrain/inception_v4/inception_v4.ckpt"

#训练好的模型参数在model文件夹下。

# 接下来可以添加的功能：
# 图像归一化：默认的是归一化到[-1,1]：(load_image/load_image.py：get_next_batch_from_path) （可以自行加一些设置参数，在此处设置）
# 需要加入模型 需修改 (train_net/train.py)
# 设置GPU使用, train_net/train.py (多GPU), main.py
# 设置学习率衰减：learningRate_1 = tf.train.exponential_decay(lr1_init, tf.subtract(global_step, 1), decay_steps, decay_rate, True)
# 加入tensorboard 可视化
# 需要修改参数更新的方法请参考：(train_net/train.py)
'''
def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)


  return optimizer'''
