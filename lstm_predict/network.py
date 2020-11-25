import tensorflow as tf


class SeqNetwork:

    def __init__(self, seq_len, hidden_num, target_len, batch_size):

        self.seq_len = seq_len
        self.hidden_num = hidden_num
        self.target_len = target_len
        self.batch_size = batch_size
        self.feature_seq = None
        self.predict_seq = None
        self.loss = None

    def setup(self, sess):

        with sess.graph.as_default():

            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.seq_len, 4],
                                        name="input")
            self.gt = tf.placeholder(dtype=tf.float32, shape=[None, self.target_len, 4],
                                     name="gt")
            self.is_train = tf.placeholder(dtype=tf.bool, name="is_train")
            
            self.feature_seq = self.featureSeqAnalysis(self.input, "seq_analysis/")

            self.predict_seq = self.predict_process(self.feature_seq)
            # self.predict_seq = self.predict_process(self.input)

            self.calculate_loss(self.predict_seq, self.gt)

    def featureSeqAnalysis(self, input, name, isHor=True):

        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            batch_num, w = shape[0],shape[1]
            print("feature_sequence_1 shape: {0} ".format(input))
            print(self.hidden_num)
            input.set_shape([None, self.seq_len, 4])
            horUnitLeft = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_num)
            # horUnitRight = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_num)

            # seqAnalysis, _ = tf.nn.bidirectional_dynamic_rnn(horUnitLeft,
            #                                                  horUnitRight, 
            #                                                  input,
            #                                                  sequence_length=[self.seq_len]*self.batch_size,
            #                                                  dtype=tf.float32)

            seqAnalysis, _ = tf.nn.dynamic_rnn(horUnitLeft,
                                               input, 
                                               sequence_length=[self.seq_len]*self.batch_size,
                                               dtype=tf.float32)
            print(seqAnalysis)
            seqAnalysis = tf.concat(seqAnalysis, axis=-1)

            seqAnalysis = tf.reshape(seqAnalysis, [-1, self.hidden_num])

            print('seqAnalysis shape {}'.format(seqAnalysis.shape))

            seqHorOut = tf.layers.dense(seqAnalysis, 4, activation=tf.nn.relu)
            seqHorOut = tf.layers.dropout(seqHorOut, rate=0.2)

            seqHorOut = tf.reshape(seqHorOut, [batch_num, self.seq_len, 4])  # batch_num * max_time
            print("seqHorOut shape {0}".format(seqHorOut.shape))

            return seqHorOut

    def predict_process(self, predict_seq):

        shape = tf.shape(predict_seq)
        batch_num, c = shape[0], shape[2]
        reshape_predict = tf.reshape(predict_seq, [batch_num*4, self.seq_len])
        predict = tf.layers.dense(inputs=reshape_predict, units=self.target_len, activation=tf.nn.relu)
        print("predict",predict)
        predict = tf.reshape(predict, [batch_num, self.target_len, 4])
        return predict
    
    def calculate_loss(self, predict, gt):

        print(predict.shape, gt.shape)
        self.loss = tf.losses.absolute_difference(gt, predict)
