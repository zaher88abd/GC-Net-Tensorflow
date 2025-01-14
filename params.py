class Params():
    def __init__(self):
        self.batch_size = 1
        self.target_h = 256
        self.target_w = 512

        self.original_h = 540
        self.original_w = 960
        self.original_c = 3

        self.max_disparity = 192

        self.enqueue_many_size = 200

    def get_input_shape(self):
        return [self.target_h, self.target_w, self.original_c]
