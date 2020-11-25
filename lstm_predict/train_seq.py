import tensorflow as tf
import os
import numpy as np
import time
import math
from tensorflow.python.client import timeline
from cfg import train_cfg as cfg
from network import SeqNetwork 
#from data_detect import *
#from data_private import *
#from data_idcard import *
from data_process.data_sequence_ohlc import get_batch
# import tensorflow.contrib.slim as slim
try:
  import cPickle as pickle
except ImportError:
  import pickle


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum',0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate',0.94,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps', 10000,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_bool('decay_staircase',True,
                          """Staircase learning rate decay by integer division""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')

class TrainSequence:

    def __init__(self, file_path):
        self.cfg        = cfg()
        self.max_iter   = self.cfg.max_iter
        self.time_seq_len = self.cfg.seq_len
        self.batch_size = self.cfg.batch_size
        self.hidden_num = self.cfg.hidden_size
        self.file_path = file_path

    def _snapshot(self, sess, iter):
        if not os.path.exists(self.cfg.OUTPUT_DIR):
            os.makedirs(self.cfg.OUTPUT_DIR)

        # Store the model snapshot
        filename = self.cfg.SNAP_SHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.cfg.OUTPUT_DIR, filename)
        self.saver.save(sess, filename, global_step=self.global_step)
        print('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = self.cfg.SNAP_SHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.cfg.OUTPUT_DIR, nfilename)

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)
        return filename, nfilename

    def debug_data_aurg(self, datas):

        pass

    def train(self, check_point=None, pkl_path=None):
        self.auto_ed = SeqNetwork(self.time_seq_len, 
                                  self.cfg.hidden_size, 
                                  self.cfg.target_len, 
                                  self.batch_size)
        with tf.Session() as sess:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.auto_ed.setup(sess)
            print("network build complete")
            self.lr = tf.train.exponential_decay(
                FLAGS.learning_rate,
                tf.train.get_global_step(),
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=FLAGS.decay_staircase,
                name='learning_rate')
            variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay,
                    self.global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            grad_op = self.optimizer.minimize(self.auto_ed.loss)
            update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            with tf.control_dependencies([variables_averages_op, grad_op, update_ops]):
                self.train_op = tf.no_op(name="train_op")
            #self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=30)
            self.saver = tf.train.Saver(max_to_keep=10)
            last_snapshot_iter = 0
            if check_point != None and pkl_path != None:
                ckp_state = tf.train.get_checkpoint_state(check_point)
                print("ckp_state.model_checkpoint_path:", 
                       ckp_state.model_checkpoint_path)
                self.saver.restore(sess, ckp_state.model_checkpoint_path)
                with open(pkl_path, 'rb') as fid:
                    last_snapshot_iter = pickle.load(fid)
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            #run_meta = tf.RunMetadata()
            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

            data_queue = get_batch(self.batch_size, self.file_path, self.cfg.seq_len)
            iteration = last_snapshot_iter + 1
            # merged = tf.summary.merge_all()
            # writer = tf.summary.FileWriter("/disk/qiujunhua/log/detect_vgg_single_branch/", sess.graph)
            while iteration < self.max_iter:
                if iteration % self.cfg.SNAP_SHOT_ITER == 0:
                    self._snapshot(sess, iteration)

                start = time.time()
                datas = next(data_queue)
                #print "data get time used: ", time.time() - start
                feed_dict = {self.auto_ed.is_train: True, 
                             self.auto_ed.input: datas[0], 
                             self.auto_ed.gt: datas[1]}
                # print(datas[0].shape)
                # print(datas[1].shape)
                start = time.time()
                lr, total_loss, predict, _ = sess.run([self.lr, 
                                                       self.auto_ed.loss, 
                                                       self.auto_ed.predict_seq, 
                                                       self.train_op], 
                                                       feed_dict=feed_dict)
                end = time.time()
                #print "sess run time used: ", end - start
                if iteration % 100 == 0:
                    print("iter: ", iteration)
                    print("total_loss: ", total_loss)
                    # print("predict: ", predict, datas[1])
                    print("lr", lr)
                # writer.add_summary(rs, iteration)
                iteration = iteration + 1


if __name__ == "__main__":
    ckp_path = None
    pkt_path = None
    net = TrainSequence("./data_process/training_data.txt")
    #ckp_path = "/disk/work/model/train_sequence/"
    #pkt_path = "/disk/work/model/train_sequence/sequence_iter_51000.pkl"
    net.train(check_point=ckp_path, pkl_path=pkt_path)
