############################################################################################################
#### Class for Deep Mem Storage Model and multiprocessing helpers
#### Author: Chris Henn
############################################################################################################

import atexit, os
from numba import njit

import numpy as np
import torch as t
import torch.multiprocessing as mp
from torch import nn as nn

from annoy import AnnoyIndex

from cuda_lib.frnn_opt_brute import frnn_cpu
import model.object_samplers as samplers
import model.object_utils as o_utils

t.manual_seed(7)



############################################################################################################
#### Deep Mem Storage Model
#       using an annoy index as memory
#       relative relationships
############################################################################################################

class Deep_Mem(nn.Module):
    '''
        Store: Builds storable image locality configurations as vectors; builds and saves an Annoy index of vectors.
        Recall: Query a configuration of image localities to retreive likely images from the Annoy index.
    '''
    def __init__(self, opt):
        super().__init__()
        self.img_size = opt.img_size
        self.center = (opt.img_size - 1) / 2
        self.train_flag = True
        self.opt = opt

        self.neb_width = 29
        self.neb_size = self.neb_width**2

        self.unfold = nn.Unfold(self.neb_width, padding=self.neb_width//2)

        self.calls = 0
        self.row_ptr = 0
        self.n_coalesce = 0

        ## set module mode for this execution run
        if self.opt.mode == 'store':
            self._forward = self.store
        elif self.opt.mode == 'recall':
            self.init_proc_pools()
            self._forward = self.recall_ann

    def forward(self, batch):
        return self._forward(batch)

    def shutdown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()

        if hasattr(self, 'proc_pool'):
            self.proc_pool.terminate()

        if hasattr(self, 'procs'):
            [proc.kill() for proc in self.procs if proc is not None]

        t.set_num_threads(2)





    def init_proc_pools(self):
        self.n_procs = os.cpu_count()

        self.manager = manager = mp.Manager()
        self.staged_q = staged_q = manager.Queue()
        self.work_q = manager.Queue()

        self.proc_pool = mp.Pool(self.n_procs)
        atexit.register(self.shutdown)

        t.set_num_threads(2)

    def sample(self, batch):

        # Create Sampler at runtime to allow saving and recalling with differently-sized images
        if not hasattr(self, 'sampler'): self.sampler = samplers.OOSampler(self.opt).to(self.opt.gpu_ids[0])

        sizes = t.zeros([1], dtype=t.int)
        data = self.sampler(batch, sizes)
        return data

    def filter_thresh(self, batch):
        thresh = 0.7
        batch[batch.le(thresh)] = 0
        batch[batch.gt(thresh)] = 1
        return batch



    ## store relationships in memory
    def store(self, batch):
        block_size = 1000000

        if not hasattr(self, 'config_stack'):
            config_stack = t.empty([block_size, self.neb_size], dtype=t.uint8)
            self.register_buffer('config_stack', config_stack)

            values = t.ones(block_size, dtype=t.long)
            self.register_buffer('values', values)

        batch = self.filter_thresh(batch)
        batch = self.unfold(batch)
        batch = batch.to(t.uint8)
        rel_vecs = batch.permute(0,2,1).reshape(-1, self.neb_size)

        ###
        row_ptr = self.row_ptr
        if row_ptr+rel_vecs.size(0) > self.config_stack.size(0):
            self.config_stack = t.cat([self.config_stack, t.empty([block_size, self.neb_size], dtype=t.int)])
            self.values = t.cat([self.values, t.ones(block_size, dtype=t.long)])

        self.config_stack[row_ptr : row_ptr+rel_vecs.size(0), :] = rel_vecs

        self.row_ptr += rel_vecs.size(0)

        self.calls += 1
        if self.calls % 200 == 0:
            self.coalesce()

    def coalesce(self):
        print('coalescing configs')
        self.config_stack = self.config_stack[:self.row_ptr, :]
        self.values = self.values[:self.row_ptr]

        self.config_stack, old_ids, new_values = self.config_stack.unique(dim=0, return_counts=True, return_inverse=True)

        if self.n_coalesce > 0:
            new_values.index_add_(0, old_ids, self.values)
        self.values = new_values

        self.row_ptr = self.values.size(0)
        self.n_coalesce += 1


    ## build annoy index with stored memory
    def build_ann(self, _):

        print('ann index build start ...')
        path = os.path.join(self.opt.save_dir, self.opt.ann_filename)
        ann_index = AnnoyIndex(self.neb_size, self.opt.ann_type)
        ann_index.on_disk_build(path)

        for i,vec in enumerate( self.config_stack.cpu().numpy() ):
            ann_index.add_item(i, vec)

        ann_index.build(self.neb_size, n_jobs=-1)
        ann_index.save(path)

        print('ann index saved at \n', path)
        print('exiting ...')
        exit(0)

    def build_rel_mp(self, batch):

        tex,pts,imgid,batch_size = self.filter_sample(batch)

        n_procs = min(batch_size.item(), self.n_procs)
        im_per_proc = ((batch_size.item()-1) // n_procs) + 1
        split_size = 812 * im_per_proc
        tex, pts, imgid = tex.split(split_size), pts.split(split_size), imgid.split(split_size)

        for i in range(len(tex)):
            self.work_q.put( (tex[i], pts[i], imgid[i]) )

        results = list()
        for i in range(n_procs):
            res = self.proc_pool.apply_async(build_rel_mp_worker,
                                             args=(self.work_q, self.staged_q, self.rel_vec_width.item(), self.lin_rad.item(), self.mem_width.item(), self.n_rolls))
            results.append(res)

        [res.wait() for res in results]

        rel_out = list()
        pts_out = list()
        for i in range(self.staged_q.qsize()):
            rel, pts = self.staged_q.get()
            rel_out.append( rel )
            pts_out.append( pts )
        rel, pts = t.cat(rel_out), t.cat(pts_out)
        return rel, pts


    ## recall from memory
    def recall_ann(self, batch):

        print("\ncompositing the response from", batch.size(0), "seeded images onto a prediction image")
        print("MAKE SURE EACH SEED IMAGE IN THIS BATCH IS THE SAME IF THE BATCH SIZE IS > 1")

        k = 40

        seed = batch.clone()
        ymask = t.arange(seed.size(2)) - self.img_size//2
        seed[:,:,  (ymask > -4) & (ymask < 6), :] = 0

        # oodl_utils.tensor_imshow(seed[0], dpi=150)

        # topil = transforms.ToPILImage()
        # seed = topil(seed[0])
        # seed = transforms.functional.affine(seed, translate=[0,0], angle=0, scale=1, shear=0, resample=PIL.Image.BICUBIC)
        # totensor = transforms.ToTensor()
        # seed = totensor(seed)[None]

        # seed = make_topbar(self.opt)
        # seed = make_nine(self.opt)
        # seed = make_circle_seed(self.opt)

        ##
        seed = self.filter_thresh(seed)
        rel_vecs = self.unfold(seed).clone()
        rel_vecs = rel_vecs.permute(0,2,1).reshape(-1, self.neb_size)
        pts = self.gen_full_grid()

        k_confs = self.query_ann_mp(rel_vecs, k)
        # k_confs = self.query_ann(rel_vecs, k)


        d_thresh_lo = 0
        d_thresh_hi = 20
        conf_thresh = 0

        configs_write, confidences_write, pts_write = self.filter_configs(k_confs, pts, d_thresh_lo, d_thresh_hi, conf_thresh)
        compt = self.render_unroll(configs_write, confidences_write, pts_write, negative=False, weight=True, avg=True)
        compt_save = compt.clone()

        compt = compt_save.clone()
        compt.div_(compt.max())
        compt.squeeze_()
        compt = t.stack([t.zeros_like(compt), t.zeros_like(compt), compt], 0)
        compt[0, ...] = seed[0, 0]
        compt.clamp_(0, 0.05)
        o_utils.tensor_imshow(compt, dpi=150)

        compt = compt_save.clone()
        compt = compt.clone().cpu().numpy()[0,0]
        compt = compt / compt.max()

        print("done")

    def query_ann_serial(self, queries, k):
        ann_path = os.path.join(self.opt.save_dir, self.opt.ann_filename)

        ann_index = AnnoyIndex(self.neb_size, self.opt.ann_type)
        ann_index.load(ann_path, prefault=False)

        k_nebs = list()
        for i, vec in enumerate(queries):
            k_nebs.append(ann_index.get_nns_by_vector(vec, k, include_distances=True))

        return k_nebs

    def query_ann_mp(self, queries, k):

        return_dict = self.manager.dict()

        query_list = queries.chunk(self.n_procs, dim=0)
        for i, chunk in enumerate(query_list):
            self.work_q.put( (str(i), chunk) )

        ann_path = os.path.join(self.opt.save_dir, self.opt.ann_filename)

        results = list()
        for i in range(self.n_procs):
            res = self.proc_pool.apply_async(query_ann_k_worker, args=(self.work_q, return_dict, k, self.neb_size, ann_path, self.opt.ann_type))
            results.append(res)
        [res.wait() for res in results]

        out = list()
        for i in range(len(return_dict)):
            out.extend( return_dict[str(i)] )
        return out




    ## Image processing
    #####

    def filter_configs(self, k_confs, pts, d_thresh_lo, d_thresh_hi, conf_thresh):

        # config_vecs = t.from_numpy( self.mem.coords.T )
        # values = t.from_numpy( self.mem.data )

        config_vecs = self.config_stack
        values = self.values

        ids_list, dists_list, pts_list = list(), list(), list()
        for i in range(len(k_confs)):
            ids, dists = k_confs[i]

            ids, dists = t.tensor(ids), t.tensor(dists)
            ids_list.append(ids); dists_list.append(dists); pts_list.append(pts[None,i,:].repeat(ids.size(0), 1))
        ids, dists, pts = t.cat(ids_list), t.cat(dists_list), t.cat(pts_list)

        ##
        dists_mask = dists.gt(d_thresh_lo) & dists.lt(d_thresh_hi)
        ids = ids[dists_mask].long()
        pts = pts[dists_mask]

        confidences = values[ids]
        confidences_mask = confidences.gt(conf_thresh)
        confidences_write = confidences[confidences_mask]

        configs = config_vecs[ids]
        configs_write = configs[confidences_mask]

        pts_write = pts[confidences_mask]

        return configs_write, confidences_write, pts_write

    def render_config(self, configs, confidences, pts, avg=False):

        pred_vals = configs[:, t.arange(0, configs.size(1), 3)]

        # pred_vals = pred_vals.mul(2).sub(1)
        pred_vals = pred_vals.mul(confidences[:,None])
        # pred_vals = pred_vals.mul(2).sub(1).mul(confidences[:,None])

        pred_vals = t.cat(pred_vals.split(1, dim=1))

        cols = t.arange(configs.size(1))
        vecs_0 = configs[:, cols[cols.fmod(3).ne(0)]]
        vecs_0 = t.cat(vecs_0.split(2, dim=1))

        vecs_0 = t.cat([t.ones_like(pts), vecs_0])

        vecs = vecs_0.float()
        vecs.div_(self.mem_width).mul_(2).sub_(1).mul_(self.lin_rad)

        locs = pts[:,:2].clone().repeat(self.n_nebs_hashed + 1, 1)
        pred_locs = locs.add(vecs)
        pred_locs[:pts.size(0), :] = pts

        ## drop padded vecs
        not_pad = vecs_0[:, 0].ne(0) | vecs_0[:, 1].ne(0)
        pred_locs = pred_locs[not_pad]
        pred_vals = pred_vals[not_pad]

        pred_locs.clamp_(0, self.img_size - 1)

        imgid = t.zeros_like(pred_locs[:, 0]).long()
        comp = o_utils.regrid(pred_vals.float(), pred_locs, imgid, self.img_size, avg=avg)
        return comp

    def render_unroll(self, configs, confidences, pts, negative=False, weight=False, avg=False):

        locs = pts[...,None].repeat(1, 1,configs.size(1))

        D1 = t.arange(self.neb_width, dtype=t.int)
        D2 = t.arange(self.neb_width, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        offsets = t.stack([gridy, gridx], 2).reshape(-1,2)
        offsets = offsets.permute(1, 0).to(pts.device)

        locs = locs.add(offsets).permute(0,2,1).reshape(-1,2)
        tex = configs.reshape(-1, 1)
        tex = tex.float()

        if negative:
            tex.mul_(2)
            tex.sub_(1)

        if weight:
            confidences = confidences[:,None].repeat(self.neb_size, 1)
            confidences = confidences.float()
            confidences.div_(confidences.max())
            tex.mul_(confidences)

        locs.sub_(self.neb_width//2)

        bounds_mask = locs[:,0].lt(self.img_size) & locs[:,1].lt(self.img_size)  &  locs[:,0].ge(0) & locs[:,1].ge(0)

        locs = locs[bounds_mask]
        tex = tex[bounds_mask]

        imgid = t.zeros_like(locs[:, 0]).long()
        comp = o_utils.regrid(tex, locs, imgid, self.img_size, avg=avg)
        return comp

    def gen_full_grid(self):
        D1 = t.arange(self.opt.img_size, dtype=t.int)
        D2 = t.arange(self.opt.img_size, dtype=t.int)
        gridy, gridx = t.meshgrid(D1, D2)
        gridy = gridy[None, None, ...].float()
        gridx = gridx[None, None, ...].float()

        pts = t.cat([gridy, gridx], 1)
        pts = pts.squeeze().permute(1,2,0)
        pts = pts.reshape(-1, pts.size(-1))
        pts = pts.repeat(self.opt.batch_size, 1)

        return pts





############################################################################################################
#### Helpers for Multiprocessing Operations
############################################################################################################

def query_ann_k_worker(work_q, return_dict, k, index_size, ann_path, ann_type):
    ann_index = AnnoyIndex(index_size, ann_type)
    ann_index.load(ann_path, prefault=False)

    while not work_q.empty():
        try:
            chunk_id, queries = work_q.get(timeout=1)
            queries = queries.numpy()

            k_nebs = list()
            for i, vec in enumerate(queries):
                k_nebs.append(ann_index.get_nns_by_vector(vec, k, include_distances=True))

            return_dict[chunk_id] = k_nebs

        except: continue


def build_rel_mp_worker(work_q, staged_q, rel_vec_width, lin_rad, mem_width, n_rolls):

    while not work_q.empty():
        try:
            data = work_q.get_nowait()
            tex,pts,imgid = data

            rel_vec = build_rel(data, rel_vec_width, lin_rad, mem_width)
            if n_rolls > 0: rel_vec = roll_rel(rel_vec, n_rolls)

            rel_vec = t.from_numpy( rel_vec )
            if n_rolls > 0: pts_out = pts.repeat(n_rolls, 1)
            else: pts_out = pts

            staged_q.put( (rel_vec, pts_out) )

        except: continue


def build_rel(data, rel_vec_width, lin_rad, mem_width):
    tex,pts,imgid = data

    # rel_vec sizes, edges, vecs, tex's
    edges = frnn_cpu.frnn_cpu(pts.numpy(), imgid.numpy(), lin_rad)
    edges = t.from_numpy(edges)
    edges = t.cat([edges, t.stack([edges[:,1], edges[:,0]], 1)], 0)

    ## vecs in rel_vec will bin to [0, mem_width)
    locs = pts[:, :2].clone()
    locs_lf, locs_rt = locs[edges[:, 0]], locs[edges[:, 1]]
    vecs = locs_rt - locs_lf
    vecs.div_(lin_rad)
    vecs.add_(1)
    vecs.div_(2)
    vecs.mul_(mem_width)
    vecs.round_()

    tex = tex.round().to(t.int32)
    tex_rt = tex[edges[:, 1]]

    args = [tex, tex_rt, vecs, edges[:, 0].contiguous()]
    args = [te.numpy() for te in args]

    rel_vec = build_rel_cpu( rel_vec_width, *args )
    return rel_vec


@njit
def build_rel_cpu(rel_vec_width, tex, tex_rt, vecs, rowids):
    write_cols = np.ones(tex.shape[0],dtype=np.int32)
    rel_vec = np.zeros( (tex.shape[0], rel_vec_width),dtype=np.int32)

    rel_vec[:,0] = tex[:,0].copy()

    n_cols = rel_vec.shape[1]
    n_vecs = vecs.shape[0]
    for vec_i in range(n_vecs):

        o_id = rowids[vec_i]
        col = write_cols[o_id]

        if col+2 >= n_cols: continue

        write_cols[o_id] += 3

        vec = vecs[vec_i]
        vec_tex = tex_rt[vec_i,0]

        rel_vec[o_id, col + 0] = vec[0]
        rel_vec[o_id, col + 1] = vec[1]
        rel_vec[o_id, col + 2] = vec_tex

    return rel_vec


def roll_rel(rel_vec, n_roll):

    rolled_rel = np.empty_like(rel_vec).repeat(n_roll, axis=0)
    for i in range(n_roll):

        roll_tmp = rel_vec.copy()
        home_tex = roll_tmp[:,0]
        roll_vecs = np.roll(roll_tmp[:, 1:], 3, axis=1)
        roll_tmp[:,0] = home_tex
        roll_tmp[:,1:] = roll_vecs

        rolled_rel[i * rel_vec.shape[0] : (i+1) * rel_vec.shape[0], :] = roll_tmp

    return rolled_rel









































