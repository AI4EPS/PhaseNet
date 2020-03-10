import tensorflow as tf
import numpy as np
import logging

def crop_and_concat(net1, net2):
  """
  the size(net1) <= size(net2)
  """
  net1_shape = net1.get_shape().as_list()
  net2_shape = net2.get_shape().as_list()
  # print(net1_shape)
  # print(net2_shape)
  # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
  size = [-1, net1_shape[1], net1_shape[2], -1]
  net2_resize = tf.slice(net2, offsets, size)
  return tf.concat([net1, net2_resize], 3)
  # else:
  #     offsets = [0, (net1_shape[1] - net2_shape[1]) // 2, (net1_shape[2] - net2_shape[2]) // 2, 0]
  #     size = [-1, net2_shape[1], net2_shape[2], -1]
  #     net1_resize = tf.slice(net1, offsets, size)
  #     return tf.concat([net1_resize, net2], 3)

def crop_only(net1, net2):
  """
  the size(net1) <= size(net2)
  """
  net1_shape = net1.get_shape().as_list()
  net2_shape = net2.get_shape().as_list()
  # print(net1_shape)
  # print(net2_shape)
  # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
  size = [-1, net1_shape[1], net1_shape[2], -1]
  net2_resize = tf.slice(net2, offsets, size)
  #return tf.concat([net1, net2_resize], 3)
  return net2_resize

class Model:
  def __init__(self, config, input_batch=None, mode='train'):
    self.depths = config.depths
    self.filters_root = config.filters_root
    self.kernel_size = config.kernel_size
    self.dilation_rate = config.dilation_rate
    self.pool_size = config.pool_size
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.n_channel = config.n_channel
    self.n_class = config.n_class
    self.class_weights = config.class_weights
    self.batch_size = config.batch_size
    self.loss_type = config.loss_type
    self.weight_decay = config.weight_decay
    self.optimizer = config.optimizer
    self.learning_rate = config.learning_rate
    self.decay_step = config.decay_step
    self.decay_rate = config.decay_rate
    self.momentum = config.momentum
    self.global_step = tf.get_variable(name="global_step", initializer=0, dtype=tf.int32)
    self.summary_train = []
    self.summary_valid = []

    self.build(input_batch, mode=mode)

  def add_placeholders(self, input_batch=None, mode="train"):
    if input_batch is None:
      self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.X_shape[0], self.X_shape[1], self.X_shape[2]], name='X')
      self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.Y_shape[0], self.Y_shape[1], self.n_class], name='y')
    else:
      self.X = input_batch[0]
      if mode in ["train", "valid", "test"]:
        self.Y = input_batch[1]
      self.input_batch = input_batch

    self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
    # self.keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    self.drop_rate = tf.placeholder(dtype=tf.float32, name="drop_rate")


  def add_prediction_op(self):
    logging.info("Model: depths {depths}, filters {filters}, "
           "filter size {kernel_size[0]}x{kernel_size[1]}, "
           "pool size: {pool_size[0]}x{pool_size[1]}, "
           "dilation rate: {dilation_rate[0]}x{dilation_rate[1]}".format(
            depths=self.depths,
            filters=self.filters_root,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            pool_size=self.pool_size))

    if self.weight_decay > 0:
      weight_decay = tf.constant(self.weight_decay, dtype=tf.float32, name="weight_constant")
      self.regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    else:
      self.regularizer = None

    self.initializer = tf.contrib.layers.xavier_initializer()

    # down sample layers
    convs = [None] * self.depths # store output of each depth

    with tf.variable_scope("Input"):
      net = self.X
      net = tf.layers.conv2d(net,
                   filters=self.filters_root,
                   kernel_size=self.kernel_size,
                   activation=None,
                   padding='same',
                   dilation_rate=self.dilation_rate,
                   kernel_initializer=self.initializer,
                   kernel_regularizer=self.regularizer,
                   #bias_regularizer=self.regularizer,
                   name="input_conv")
      net = tf.layers.batch_normalization(net,
                        training=self.is_training,
                        name="input_bn")
      net = tf.nn.relu(net,
               name="input_relu")
      # net = tf.nn.dropout(net, self.keep_prob)
      net = tf.layers.dropout(net,
                  rate=self.drop_rate,
                  training=self.is_training,
                  name="input_dropout")


    for depth in range(0, self.depths):
      with tf.variable_scope("DownConv_%d" % depth):
        filters = int(2**(depth) * self.filters_root)

        net = tf.layers.conv2d(net,
                     filters=filters,
                     kernel_size=self.kernel_size,
                     activation=None,
                     use_bias=False,
                     padding='same',
                     dilation_rate=self.dilation_rate,
                     kernel_initializer=self.initializer,
                     kernel_regularizer=self.regularizer,
                     #bias_regularizer=self.regularizer,
                     name="down_conv1_{}".format(depth + 1))
        net = tf.layers.batch_normalization(net,
                          training=self.is_training,
                          name="down_bn1_{}".format(depth + 1))
        net = tf.nn.relu(net,
                 name="down_relu1_{}".format(depth+1))
        net = tf.layers.dropout(net,
                    rate=self.drop_rate,
                    training=self.is_training,
                    name="down_dropout1_{}".format(depth + 1))

        convs[depth] = net

        if depth < self.depths - 1:
          net = tf.layers.conv2d(net,
                       filters=filters,
                       kernel_size=self.kernel_size,
                       strides=self.pool_size,
                       activation=None,
                       use_bias=False,
                       padding='same',
                       dilation_rate=self.dilation_rate,
                       kernel_initializer=self.initializer,
                       kernel_regularizer=self.regularizer,
                       #bias_regularizer=self.regularizer,
                       name="down_conv3_{}".format(depth + 1))
          net = tf.layers.batch_normalization(net,
                            training=self.is_training,
                            name="down_bn3_{}".format(depth + 1))
          net = tf.nn.relu(net,
                   name="down_relu3_{}".format(depth+1))
          net = tf.layers.dropout(net,
                    rate=self.drop_rate,
                    training=self.is_training,
                    name="down_dropout3_{}".format(depth + 1))


    # up layers
    for depth in range(self.depths - 2, -1, -1):
      with tf.variable_scope("UpConv_%d" % depth):
        filters = int(2**(depth) * self.filters_root)
        net = tf.layers.conv2d_transpose(net,
                         filters=filters,
                         kernel_size=self.kernel_size,
                         strides=self.pool_size,
                         activation=None,
                         use_bias=False,
                         padding="same",
                         kernel_initializer=self.initializer,
                         kernel_regularizer=self.regularizer,
                         #bias_regularizer=self.regularizer,
                         name="up_conv0_{}".format(depth+1))
        net = tf.layers.batch_normalization(net,
                          training=self.is_training,
                          name="up_bn0_{}".format(depth + 1))
        net = tf.nn.relu(net,
                 name="up_relu0_{}".format(depth+1))
        net = tf.layers.dropout(net,
                    rate=self.drop_rate,
                    training=self.is_training,
                    name="up_dropout0_{}".format(depth + 1))

        
        #skip connection
        net = crop_and_concat(convs[depth], net)
        #net = crop_only(convs[depth], net)

        net = tf.layers.conv2d(net,
                     filters=filters,
                     kernel_size=self.kernel_size,
                     activation=None,
                     use_bias=False,
                     padding='same',
                     dilation_rate=self.dilation_rate,
                     kernel_initializer=self.initializer,
                     kernel_regularizer=self.regularizer,
                     #bias_regularizer=self.regularizer,
                     name="up_conv1_{}".format(depth + 1))
        net = tf.layers.batch_normalization(net,
                          training=self.is_training,
                          name="up_bn1_{}".format(depth + 1))
        net = tf.nn.relu(net,
                 name="up_relu1_{}".format(depth + 1))
        net = tf.layers.dropout(net,
                    rate=self.drop_rate,
                    training=self.is_training,
                    name="up_dropout1_{}".format(depth + 1))


    # Output Map
    with tf.variable_scope("Output"):
      net = tf.layers.conv2d(net,
                   filters=self.n_class,
                   kernel_size=(1,1),
                   activation=None,
                   padding='same',
                   #dilation_rate=self.dilation_rate,
                   kernel_initializer=self.initializer,
                   kernel_regularizer=self.regularizer,
                   #bias_regularizer=self.regularizer,
                   name="output_conv")
      # net = tf.nn.relu(net,
      #                     name="output_relu")
      # net = tf.layers.dropout(net,
      #                         rate=self.drop_rate,
      #                         training=self.is_training,
      #                         name="output_dropout")
      # net = tf.layers.batch_normalization(net,
      #                                    training=self.is_training,
      #                                    name="output_bn")
      output = net
     
    with tf.variable_scope("representation"):
      self.representation = convs[-1]

    with tf.variable_scope("logits"):
      self.logits = output
      tmp = tf.summary.histogram("logits", self.logits)
      self.summary_train.append(tmp)

    with tf.variable_scope("preds"):
      self.preds = tf.nn.softmax(output)
      tmp = tf.summary.histogram("preds", self.preds)
      self.summary_train.append(tmp)

  def add_loss_op(self):
    if self.loss_type == "cross_entropy":
      with tf.variable_scope("cross_entropy"):
        flat_logits = tf.reshape(self.logits, [-1, self.n_class], name="logits")
        flat_labels = tf.reshape(self.Y, [-1, self.n_class], name="labels")
        if (np.array(self.class_weights) != 1).any():
          class_weights = tf.constant(np.array(self.class_weights, dtype=np.float32), name="class_weights")
          weight_map = tf.multiply(flat_labels, class_weights)
          weight_map = tf.reduce_sum(weight_map, axis=1)
          loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                     labels=flat_labels)
#                     loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits,
#                                                                       labels=flat_labels)
          weighted_loss = tf.multiply(loss_map, weight_map)
          loss = tf.reduce_mean(weighted_loss)
        else:
          loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                         labels=flat_labels))
#                     loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits,
#                                                                                   labels=flat_labels))
    elif self.loss_type == "IOU":
      with tf.variable_scope("IOU"):
        eps = 1e-7
        loss = 0
        for i in range(1, self.n_class): 
          intersection = eps + tf.reduce_sum(self.preds[:,:,:,i] * self.Y[:,:,:,i], axis=[1,2])
          union = eps + tf.reduce_sum(self.preds[:,:,:,i], axis=[1,2]) + tf.reduce_sum(self.Y[:,:,:,i], axis=[1,2]) 
          loss += 1 - tf.reduce_mean(intersection / union)
    elif self.loss_type == "mean_squared":
      with tf.variable_scope("mean_squared"):
        flat_logits = tf.reshape(self.logits, [-1, self.n_class], name="logits")
        flat_labels = tf.reshape(self.Y, [-1, self.n_class], name="labels")
        with tf.variable_scope("mean_squared"):
          loss = tf.losses.mean_squared_error(labels=flat_labels, predictions=flat_logits) 
    else:
      raise ValueError("Unknown loss function: " % self.loss_type)

    tmp = tf.summary.scalar("train_loss", loss)
    self.summary_train.append(tmp)
    tmp = tf.summary.scalar("valid_loss", loss)
    self.summary_valid.append(tmp)

    if self.weight_decay > 0:
      with tf.name_scope('weight_loss'):
        tmp = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        weight_loss = tf.add_n(tmp, name="weight_loss")
      self.loss = loss + weight_loss 
    else:
      self.loss = loss 

  def add_training_op(self):
    if self.optimizer == "momentum":
      self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                 global_step=self.global_step,
                                 decay_steps=self.decay_step,
                                 decay_rate=self.decay_rate,
                                 staircase=True)
      optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node,
                           momentum=self.momentum)
    elif self.optimizer == "adam":
      self.learning_rate_node = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                 global_step=self.global_step,
                                 decay_steps=self.decay_step,
                                 decay_rate=self.decay_rate,
                                 staircase=True)

      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
    tmp = tf.summary.scalar("learning_rate", self.learning_rate_node)
    self.summary_train.append(tmp)

  def add_metrics_op(self):
    with tf.variable_scope("metrics"):

      Y= tf.argmax(self.Y, -1)
      confusion_matrix = tf.cast(tf.confusion_matrix(
          tf.reshape(Y, [-1]), 
          tf.reshape(self.preds, [-1]), 
          self.n_class, name='confusion_matrix'),
          dtype=tf.float32)

      # with tf.variable_scope("P"):
      c = tf.constant(1e-7, dtype=tf.float32)
      precision_P =  (confusion_matrix[1,1] + c) / (tf.reduce_sum(confusion_matrix[:,1]) + c)
      recall_P = (confusion_matrix[1,1] + c) / (tf.reduce_sum(confusion_matrix[1,:]) + c)
      f1_P = 2 * precision_P * recall_P / (precision_P + recall_P)

      tmp1 = tf.summary.scalar("train_precision_p", precision_P)
      tmp2 = tf.summary.scalar("train_recall_p", recall_P)
      tmp3 = tf.summary.scalar("train_f1_p", f1_P)
      self.summary_train.extend([tmp1, tmp2, tmp3])

      tmp1 = tf.summary.scalar("valid_precision_p", precision_P)
      tmp2 = tf.summary.scalar("valid_recall_p", recall_P)
      tmp3 = tf.summary.scalar("valid_f1_p", f1_P)
      self.summary_valid.extend([tmp1, tmp2, tmp3])

      # with tf.variable_scope("S"):
      precision_S =  (confusion_matrix[2,2] + c) / (tf.reduce_sum(confusion_matrix[:,2]) + c)
      recall_S = (confusion_matrix[2,2] + c) / (tf.reduce_sum(confusion_matrix[2,:]) + c)
      f1_S = 2 * precision_S * recall_S / (precision_S + recall_S)

      tmp1 = tf.summary.scalar("train_precision_s", precision_S)
      tmp2 = tf.summary.scalar("train_recall_s", recall_S)
      tmp3 = tf.summary.scalar("train_f1_s", f1_S)
      self.summary_train.extend([tmp1, tmp2, tmp3])

      tmp1 = tf.summary.scalar("valid_precision_s", precision_S)
      tmp2 = tf.summary.scalar("valid_recall_s", recall_S)
      tmp3 = tf.summary.scalar("valid_f1_s", f1_S)
      self.summary_valid.extend([tmp1, tmp2, tmp3])
      
      self.precision = [precision_P, precision_S]
      self.recall = [recall_P, recall_S]
      self.f1 = [f1_P, f1_S]



  def train_on_batch(self, sess, inputs_batch, labels_batch, summary_writer, drop_rate=0.0):
    feed = {self.X: inputs_batch,
            self.Y: labels_batch,
            self.drop_rate: drop_rate,
            self.is_training: True}
            

    _, step_summary, step, loss = sess.run([self.train_op,
                                            self.summary_train,
                                            self.global_step,
                                            self.loss],
                                            feed_dict=feed)
    summary_writer.add_summary(step_summary, step)
    return loss

  def valid_on_batch(self, sess, inputs_batch, labels_batch, summary_writer):
    feed = {self.X: inputs_batch,
            self.Y: labels_batch,
            self.drop_rate: 0,
            self.is_training: False}
            
    step_summary, step, loss, preds = sess.run([self.summary_valid,
                                                self.global_step,
                                                self.loss,
                                                self.preds],
                                                feed_dict=feed)
    summary_writer.add_summary(step_summary, step)
    return loss, preds

  # def train_on_batch(self, sess, summary_writer, drop_rate=0.0, raw_data=False):
  #   feed = {self.drop_rate: drop_rate,
  #           self.is_training: True}
  #   if raw_data:
  #     _, step_summary, step, loss, preds, logits, \
  #     X_batch, Y_batch = sess.run([self.train_op, 
  #                                  self.summary_train,
  #                                  self.global_step,
  #                                  self.loss,
  #                                  self.preds,
  #                                  self.logits,
  #                                  self.X,
  #                                  self.Y],
  #                                  feed_dict=feed)
  #     summary_writer.add_summary(step_summary, step)
  #     return loss, preds, logits, X_batch, Y_batch
  #   else:
  #     _, step_summary, step, loss = sess.run([self.train_op,
  #                         self.summary_train,
  #                         self.global_step,
  #                         self.loss],
  #                         feed_dict=feed)
  #     summary_writer.add_summary(step_summary, step)
  #     return loss

  def test_on_batch(self, sess, summary_writer):
    feed = {self.drop_rate: 0,
            self.is_training: False}
    step_summary, step, loss, preds, \
    X_batch, Y_batch, fname_batch, \
    itp_batch, its_batch = sess.run([self.summary_valid,
                                     self.global_step,
                                     self.loss,
                                     self.preds,
                                     self.X,
                                     self.Y,
                                     self.input_batch[2],
                                     self.input_batch[3],
                                     self.input_batch[4]],
                                     feed_dict=feed)
    summary_writer.add_summary(step_summary, step)
    return loss, preds, X_batch, Y_batch, fname_batch, itp_batch, its_batch


  def build(self, input_batch=None, mode='train'):
    self.add_placeholders(input_batch, mode)
    self.add_prediction_op()
    if mode in ["train", "valid", "test"]:
      self.add_loss_op()
      self.add_training_op()
      # self.add_metrics_op()
      self.summary_train = tf.summary.merge(self.summary_train)
      self.summary_valid = tf.summary.merge(self.summary_valid)
    return 0
