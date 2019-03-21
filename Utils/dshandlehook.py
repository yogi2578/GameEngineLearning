import tensorflow as tf

class DSHandleHook(tf.train.SessionRunHook):
    def __init__(self, train_str, valid_str):
        """
        helper class for swiching multi threaded input queue pipelines in graph during run time using feedable iterator with state
        """
        self.train_str = train_str
        self.valid_str = valid_str
        self.train_handle = None
        self.valid_handle = None

    def after_create_session(self, session, coord):
        del coord
        if self.train_str is not None:
            self.train_handle, self.valid_handle = session.run([self.train_str,
                                                                self.valid_str])
        print('session run ds string-handle done....')