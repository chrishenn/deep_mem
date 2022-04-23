import time, os, sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from model.oodl_vis import OODL_Vis
from plotter.plot_pretty import plot_pretty
from options.options import Options_Parent


class Options(Options_Parent):

    def initialize(self, parser):

        parser.add_argument('--exp_name',   type=str,   default='[cf32|te rot] oodl1[32-256|r1248|pool rt->lf]', help=' ')

        # training options
        parser.add_argument('--gpu_ids',    type=str, default='0', help='')

        parser.add_argument('--n_epochs',  type=int,  default=11, help='number of training epochs')

        # I/O options
        parser.add_argument('--vis_network', type=bool, default=True, help='set True for loss visualization')
        parser.add_argument('--vis_file',    type=bool, default=True, help='')

        self.parser = parser
        return self.parser

def gen_from_datafile():
    opt = Options().parse()
    vis = OODL_Vis(opt)

    with open('/home/chris/Documents/oodl_local/perf/[cf32|te rot] best.txt', 'r') as file:
        for i,line in enumerate(file):
            line = line.split()

            epoch = i

            train_loss = float( line[1] ) / 50
            test_loss = float( line[2] ) / 50

            train_top1 = 100. - float( line[3] )
            test_top1 = 100. - float( line[4] )

            vis.vis_draw([epoch, train_loss, train_top1, test_loss, test_top1])
            time.sleep(1)

def gen_pretty_from_datafile():
    n_epoch = 11

    filenames = ['/home/chris/Documents/oodl_local/graphs/OODL_2/tn64_oodl2_rot.txt' ]

    # dirname = '/home/chris/Documents/oodl_local/graphs/DCONV/'
    # for filename in glob.glob(dirname + '*.txt'):

    for filename in filenames:
        train_acc, val_acc = np.empty(n_epoch), np.empty(n_epoch)
        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                line = line.split()

                # train_loss = float(line[1]) / 50
                # test_loss = float(line[2]) / 50

                train_acc[i] = 100. - float( line[3] )
                val_acc[i] = 100. - float( line[4] )

        plot_pretty([train_acc, val_acc], ['train_acc', 'val_acc'], xlabel='Epoch', ylabel='Accuracy', ylim=[0, 100])
        plt.savefig(filename.split('.')[0] + '.png', pad_inches=0)
        plt.clf()

if __name__ == '__main__':
    gen_pretty_from_datafile()
