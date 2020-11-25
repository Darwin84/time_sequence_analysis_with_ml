from data_utils import *
import time
import numpy as np
import random


class DataSequence:

    def __init__(self, file_path, batch_size, seq_len):

        self.file_path = file_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_data = []
        self.train_index = 0

    def _shuffle_dataset(self, data_set):
        random.shuffle(data_set)

    def construct_train_data(self):

        with open(self.file_path, "r") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.strip()
                line = line[1:-1]
                line = line.split(",")
                data = list(map(eval, line))
                self.train_data.append(data)

    def data_forward(self, data_list, data_index):
        
        current_label_list = []
        traing_data_array = np.zeros([self.batch_size, self.seq_len, 4])
        for i in range(self.batch_size):
            if data_index >= len(data_list):
                self._shuffle_dataset(data_list)
                data_index = 0
            current_data_list = []
            # print(len(data_list[data_index]))
            current_data_list.extend(data_list[data_index][:-1])
            current_label_list.append([data_list[data_index][-1]])
            data_index += 1
            current_data_array = np.array(current_data_list)
            current_data_open = current_data_array[::4]
            # print(current_data_open.shape)
            current_data_high = current_data_array[1::4]
            # print(current_data_high.shape)
            current_data_low = current_data_array[2::4]
            current_data_close = current_data_array[3::4]
        
            # print(traing_data_array.shape)
            traing_data_array[i,:,0] = current_data_open
            traing_data_array[i,:,1] = current_data_high
            traing_data_array[i,:,2] = current_data_low
            traing_data_array[i,:,3] = current_data_close
        current_label_array = np.array(current_label_list)
        current_label_array = current_label_array[:, :, np.newaxis]
        # print(current_data_array.shape, current_label_array.shape)
        return traing_data_array, current_label_array, data_index

    def train_data_forward(self):

        while True:

            data_list, label_list, self.train_index = self.data_forward(self.train_data, self.train_index)

            yield data_list, label_list


def get_batch(batch_size, file_path, seq_len, num_workers=1):

    data_layer = DataSequence(file_path, batch_size, seq_len)
    data_layer.construct_train_data()
    enqueuer = None
    try:
        enqueuer = GeneratorEnqueuer(data_layer.train_data_forward(), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == "__main__":

    file_path = "./training_data.txt"
    data_next = get_batch(1, file_path, 51)
    for i in range(1):
        data, label = next(data_next)
        print(data.shape, label.shape)