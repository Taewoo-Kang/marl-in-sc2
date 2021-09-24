import logging
import tensorboard_logger
import os


class Logger:
    def __init__(self, args):
        self.console_logger = self._init_console_logger()
        self.tb_logger = self._init_tb_logger(args)

    def _init_console_logger(self):
        console_logger = logging.getLogger()
        console_logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        console_logger.addHandler(ch)
        console_logger.setLevel('DEBUG')
        return console_logger

    def _init_tb_logger(self, args):
        file_path = "{}/{}_{}".format(args.map_name, args.algorithm, args.start_time)
        tb_logs_dir = os.path.join(args.save_dir, "tb_logs", file_path)
        tensorboard_logger.configure(tb_logs_dir)
        return tensorboard_logger

    def log(self, key, value, t):
        self.tb_logger.log_value(key, value, t)
        log_str = "{} {} {}".format(key, value, t)
        self.console_logger.info(log_str)
