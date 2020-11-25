class train_cfg:
    def __init__(self):
        self.SNAP_SHOT_ITER = 1000
        self.DISPLAY_ITER = 100
        self.LEARNING_RATE = 0.0001
        self.max_iter = 100000
        self.OUTPUT_DIR = "/disk/work/model/train_sequence/"
        self.SNAP_SHOT_PREFIX= "sequence"
        self.need_pretrained = True
        self.TEST_ITER = 5000
        self.hidden_size = 256
        self.seq_len = 20
        self.target_len = 1
        self.batch_size = 128