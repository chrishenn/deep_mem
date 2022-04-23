import matplotlib.pyplot as plt
import torch as t
import numpy as np

from model.dsepconvnet import dscnet
from model.oonet_DEV import OONet_local
from train_oonet import get_loader
from model import meter
from model.meter import AverageMeter, ProgressMeter

import os, sys, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plotter.plot_pretty import plot_pretty
from options.options import Options_Parent

checkpoint_dir = '/home/chris/Documents/oodl_local/checkpoints/'


class Options(Options_Parent):

    def initialize(self, parser):

        parser.add_argument('--model_name', type=str, default='OONet_local', help='chooses which oonet to use.')
        # parser.add_argument('--model_name', type=str, default='dscnet', help='chooses which oonet to use.')
        parser.add_argument('--feat_sizes', type=list, default=[3,32,64,128,256], help='feature sizes')
        parser.add_argument('--exp_name',   type=str,   default='', help=' ')

        # data options
        # parser.add_argument('--dataset_mode', type=str, default='tiny-imagenet', help='chooses what datasets are loaded.')
        parser.add_argument('--dataset_mode', type=str, default='cifar', help='chooses what datasets are loaded.')

        parser.add_argument('--data_dir',     type=str, default='/home/chris/Documents/oodl_local/data/', help='chooses what datasets are loaded.')
        parser.add_argument('--img_size',     type=int, default=64, help='square image; integer size for one side')

        parser.add_argument('--train_datamode', type=str,  default='train', help='train data mode: set to "train" for training-set; set to "test" for test set')
        parser.add_argument('--train_size',     type=int,  default=None, help='specify dataset size in images - at least 1/30th of the full size. Set "None" for full size.')
        parser.add_argument('--train_rotate',   type=bool, default=False, help='set True to randomly-rotate train images')
        parser.add_argument('--train_scale',    type=float,default=False, help='randomly scale train images in given range')
        parser.add_argument('--train_normalize',type=bool, default=True, help='normalize train images to range=(-1,1)')
        parser.add_argument('--train_shuffle',  type=bool, default=True, help='random-shuffle train images')

        parser.add_argument('--test_datamode',  type=str,   default='test', help='test data mode: set to "train" for training-set; set to "test" for test set')
        parser.add_argument('--test_size',      type=int,   default=None, help='')
        parser.add_argument('--test_rotate',    type=bool,  default=False, help='set True to randomly-rotate test images')
        parser.add_argument('--test_scale',     type=float, default=False, help='randomly scale test images in given range')
        parser.add_argument('--test_normalize', type=bool,  default=True, help='normalize test images to range=(-1,1)')
        parser.add_argument('--test_shuffle',   type=bool,  default=True, help='random-shuffle test images')

        # training options
        parser.add_argument('--gpu_ids',   type=str, default='1', help='')
        parser.add_argument('--n_threads', type=int, default=2, help='')

        parser.add_argument('--n_epochs',  type=int,  default=30, help='number of training epochs')

        parser.add_argument('--batch_size', type=int, default=40, help='batch size')

        parser.add_argument('--profile', type=bool, default=False, help='set True to run a profiling epoch')

        parser.add_argument('--classifier', type=bool, default=True,  help='classifier ')
        parser.add_argument('--resnet',     type=bool, default=False,  help=' ')
        parser.add_argument('--debug',      type=bool, default=False, help='debug ')
        parser.add_argument('--model_parallel', type=bool, default=False, help='')

        self.parser = parser
        return self.parser


def gen_arr_from_weights():
    opt = Options().parse()
    t.cuda.set_device(opt.gpu_ids[0])
    t.set_grad_enabled(False)

    opt.test_loader = get_loader(opt, False)
    model = model = oodl_model.OODL_Model(opt).requires_grad_(False)

    val_acc = t.empty(opt.n_epochs)
    test_meter = AverageMeter('Acc@1', ':6.2f')
    test_progress = ProgressMeter(len(opt.test_loader), test_meter, prefix='test: ')

    # dirname = checkpoint_dir + 'cf32_dconvGlob_weights1-30/'
    # dirname = checkpoint_dir + 'tn64_dconvGlob_weights1-30/'
    # dirname = checkpoint_dir + 'cf32_oodl_weights1-30/'
    dirname = checkpoint_dir + 'cf64_oodl_weights1-30/'
    # dirname = checkpoint_dir + 'tn64_oodl_weights1-30/'
    filenames = sorted( zip( [int( file.split('_')[1] ) for file in os.listdir(dirname)], os.listdir(dirname) ) )
    filenames = [file[1] for file in filenames]
    for epoch, filename in enumerate( filenames ):
        test_meter.reset()

        print("loading from ", filename)
        model.cpu()
        model.load_state_dict(t.load(dirname + filename))
        model.cuda()

        for i, data in enumerate(opt.test_loader):
            inputs, labels = data
            inputs, labels = tuple(te.cuda() for te in inputs) if isinstance(inputs, tuple) else inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            acc1 = meter.accuracy(outputs, labels, topk=(1,))
            test_meter.update(acc1[0].item(), opt.batch_size)

            if (i+1)%100 == 0 or (i+1) == len(opt.test_loader):
                test_progress.print(i+1, epoch)

        val_acc[epoch] = test_meter.get_avg()

    # saveas = checkpoint_dir + 'tn64_oodl_test_rot.arr'
    # saveas = checkpoint_dir + 'tn64_oodl_test_scale.arr'
    # saveas = checkpoint_dir + 'tn64_oodl_test_none.arr'
    # saveas = checkpoint_dir + 'cf32_oodl_test_rot.arr'
    # saveas = checkpoint_dir + 'cf32_oodl_test_scale.arr'
    # saveas = checkpoint_dir + 'cf32_oodl_test_none.arr'
    # saveas = checkpoint_dir + 'cf64_oodl_test_rot.arr'
    # saveas = checkpoint_dir + 'cf64_oodl_test_scale.arr'
    saveas = checkpoint_dir + 'cf64_oodl_test_none.arr'

    # saveas = checkpoint_dir + 'tn64_dconv_test_rot.arr'
    # saveas = checkpoint_dir + 'tn64_dconv_test_scale.arr'
    # saveas = checkpoint_dir + 'tn64_dconv_test_none.arr'
    # saveas = checkpoint_dir + 'cf32_dconv_test_rot.arr'
    # saveas = checkpoint_dir + 'cf32_dconv_test_scale.arr'
    # saveas = checkpoint_dir + 'cf32_dconv_test_none.arr'

    print("saving at: ", saveas)
    t.save(val_acc, saveas)



def perf_array_fromrunfile_train():

    train_acc = t.empty(30)
    written = 0

    filename = '/home/chris/Documents/oodl_local/checkpoints/cf64_oodl_scale_run.txt'

    saveas = '/home/chris/Documents/oodl_local/checkpoints/cf64_oodl_train.arr'

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.split()

        if len(line) > 4 and line[2] == 'train:' and int(line[4].strip('/')) == int(line[5].strip(']')):
            train_acc[written] = float( line[-1].strip(')') )
            written += 1

    t.save(train_acc, saveas)

def perf_array_fromrunfile_test():

    test_acc = t.empty(30)
    written = 0

    filename = '/home/chris/Documents/oodl_local/checkpoints/cf64_oodl_scale_run.txt'

    saveas = '/home/chris/Documents/oodl_local/checkpoints/cf64_oodl_test_scale.arr'

    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.split()

        if len(line) > 4 and line[2] == 'TEST:' and int( line[4].strip('/') ) == int( line[5].strip(']') ):

            test_acc[written] = float(line[-1].strip(')'))
            written += 1

    t.save(test_acc, saveas)

def gen_pretty_from_arr():

    # train_file = checkpoint_dir + 'cf32_dconv_train.arr'
    # train_file = checkpoint_dir + 'tn64_dconv_train.arr'

    # train_file = checkpoint_dir + 'cf32_oodl_train.arr'
    train_file = checkpoint_dir + 'cf64_oodl_train.arr'
    # train_file = checkpoint_dir + 'tn64_oodl_train.arr'

    #####
    # test_file = checkpoint_dir + 'cf32_dconv_test_rot.arr'
    # test_file = checkpoint_dir + 'cf32_dconv_test_scale.arr'
    # test_file = checkpoint_dir + 'cf32_dconv_test_none.arr'

    # test_file = checkpoint_dir + 'tn64_dconv_test_rot.arr'
    # test_file = checkpoint_dir + 'tn64_dconv_test_scale.arr'
    # test_file = checkpoint_dir + 'tn64_dconv_test_none.arr'

    # test_file = checkpoint_dir + 'cf32_oodl_test_rot.arr'
    # test_file = checkpoint_dir + 'cf32_oodl_test_scale.arr'
    # test_file = checkpoint_dir + 'cf32_oodl_test_none.arr'

    # test_file = checkpoint_dir + 'cf64_oodl_test_rot.arr'
    # test_file = checkpoint_dir + 'cf64_oodl_test_scale.arr'
    test_file = checkpoint_dir + 'cf64_oodl_test_none.arr'

    # test_file = checkpoint_dir + 'tn64_oodl_test_rot.arr'
    # test_file = checkpoint_dir + 'tn64_oodl_test_scale.arr'
    # test_file = checkpoint_dir + 'tn64_oodl_test_none.arr'

    # save_dir = '/home/chris/Documents/oodl_local/graphs/DCONV/'
    # saveas = save_dir + 'cf32[30]_DCONV_rot.png'
    # saveas = save_dir + 'cf32[30]_DCONV_scale.png'
    # saveas = save_dir + 'cf32[30]_DCONV_none.png'

    # saveas = save_dir + 'tn64[30]_DCONV_rot.png'
    # saveas = save_dir + 'tn64[30]_DCONV_scale.png'
    # saveas = save_dir + 'tn64[30]_DCONV_none.png'

    save_dir = '/home/chris/Documents/oodl_local/graphs/OODL_2/'
    # saveas = save_dir + 'cf32[30]_OODL_rot.png'
    # saveas = save_dir + 'cf32[30]_OODL_scale.png'
    # saveas = save_dir + 'cf32[30]_OODL_none.png'

    # saveas = save_dir + 'cf64[30]_OODL_rot.png'
    # saveas = save_dir + 'cf64[30]_OODL_scale.png'
    saveas = save_dir + 'cf64[30]_OODL_none.png'

    # saveas = save_dir + 'tn64[30]_OODL_rot.png'
    # saveas = save_dir + 'tn64[30]_OODL_scale.png'
    # saveas = save_dir + 'tn64[30]_OODL_none.png'

    train_acc = t.load(train_file).numpy()
    test_acc = t.load(test_file).numpy()

    plot_pretty([train_acc, test_acc], ['train_acc', 'val_acc'], xlabel='Epoch', ylabel='Accuracy', ylim=[0, 100])
    # plt.show()
    plt.savefig(saveas , pad_inches=0)


def bump_iter_names():
    dirname = '/home/chris/Documents/oodl_local/checkpoints/cf64_oodl_weights21-31/'

    add_val = 30000

    for filename in glob.glob(dirname + '*'):
        list = os.path.split(filename)[1]
        list = list.split('_')
        list[1] = str( int(list[1]) + add_val )
        new_filename = '_'.join(list)
        os.rename(filename, dirname + new_filename)


def rename_in_folder():
    dirname = '/home/chris/Documents/oodl_local/graphs/grouper/truck_groups/notrain/'

    for filename in glob.glob(dirname + '*'):
        dir, fname = os.path.split(filename)

        new_fname = fname.replace('trkGrp_notrain_', '')

        os.rename(filename, os.path.join(dir, new_fname) )

if __name__ == '__main__':
    # gen_arr_from_weights()
    # perf_array_fromrunfile_train()
    # perf_array_fromrunfile_test()
    # gen_pretty_from_arr()

    # bump_iter_names()
    rename_in_folder()
























