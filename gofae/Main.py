import sys
import argparse
import torch
from core.GoFAE import GoFAE
from utilities.IOUtilities import eprint
import yaml
import numpy as np
import random
from utilities.IOUtilities import tuple_constructor


class Main(object):

    @classmethod
    def init(cls, cmd_args, tee):
        if tee is None:
            if sys.platform == "win32":
                tee = open("NUL:")
            else:
                #  should be fine for OSX/Linux
                tee = open("/dev/null")

        parser = Main.getParser()
        args = parser.parse_args(cmd_args[1:])

        yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

        with open(args.filename, 'r') as file:
            try:
                config = yaml.load(file, Loader=yaml.SafeLoader)
            except yaml.YAMLError as exc:
                print(exc)

        cls.seed = config['logging_params']['seed']
        cls.reprod = config['logging_params']['reprod']
        cls.fQuiet = config['trainer_params']['fQuiet']  # args.fQuiet

        if cls.reprod:
            torch.manual_seed(cls.seed)
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(cls.seed)
            np.random.seed(cls.seed)
            random.seed(cls.seed)

        if not cls.fQuiet:
            eprint("args =")
            for arg in cmd_args:
                eprint(" " + arg)
            eprint()
            eprint()

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not cls.fQuiet:
            print('Using device:', device, flush=True)
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0), flush=True)
        return GoFAE(device, config)

    @classmethod
    def getParser(cls):
        parser = argparse.ArgumentParser(
            description='Implementation of the Goodness of Fit Autoencoder.')
        parser.add_argument('--config', '-c',
                            dest="filename",
                            metavar='FILE',
                            help='path to the config file',
                            default='configs/example.yaml')
        return parser

    @classmethod
    def main(cls, cmd_args):
        gofae = cls.init(cmd_args, sys.stdout)
        if gofae.train:
            gofae.run()
        else:
            gofae.restore_eval()


if __name__ == '__main__':
    Main.main(sys.argv)
