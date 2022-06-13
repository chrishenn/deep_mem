from unittest import TestCase
import torch as t
import os

t.ops.load_library(os.path.split(__file__)[0] + "/cuda_lib/frnn_opt_brute/build/libfrnn_ts.so")

def get_objects(sizes):

    b_size, c_size, y_size, x_size = sizes

    D1 = t.arange(y_size, dtype=t.int)
    D2 = t.arange(x_size, dtype=t.int)

    gridy, gridx = t.meshgrid(D1,D2)
    gridy = gridy.unsqueeze(0).unsqueeze(0).repeat(b_size,1,1,1).float()
    gridx = gridx.unsqueeze(0).unsqueeze(0).repeat(b_size,1,1,1).float()
    zeros = t.zeros_like(gridx)
    ones =  t.ones_like(gridx)
    angles = t.zeros_like(gridx)

    geom = t.cat([gridy, gridx, zeros, angles, ones, zeros], 1)

    imgid = t.arange(b_size).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1, y_size, x_size)

    geom =    geom.permute([0,2,3,1])
    imgid =   imgid.permute([0,2,3,1])

    geom =    geom.reshape(-1,geom.size(3))
    imgid =   imgid.reshape(-1,imgid.size(3))
    imgid =   imgid.squeeze(1)

    return geom, imgid

def gen_data():
    batch_size = t.tensor(10).cuda()
    pts, imgid = get_objects([10, 3, 64, 64])
    pts, imgid = pts.cuda(), imgid.cuda()
    lin_radius = t.tensor(2.0).cuda()
    scale_radius = t.tensor(1.0).cuda()

    return pts, imgid, lin_radius, scale_radius, batch_size

class test_frnn(TestCase):

    def test_1(self):
        data = gen_data()
        data = t.ops.my_ops.frnn_ts_kernel(*data)[0]
        print(data.shape)
        self.assertIsInstance(data, t.Tensor)


