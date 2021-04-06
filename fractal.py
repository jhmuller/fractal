
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import warnings
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utilities import Utilities


class ImageUtils(object):

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def spatial_shrink(block, spatial_factor):
        result = np.zeros((block.shape[0] // spatial_factor,
                           block.shape[1] // spatial_factor))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                try:
                    result[i, j] = np.mean(block[i * spatial_factor:(i + 1) * spatial_factor,
                                       j * spatial_factor:(j + 1) * spatial_factor])
                except Exception as e:
                    msg = Utilities.last_exception_info()
                    warnings.warn(msg)
                    raise RuntimeError(msg)
        return result

    @staticmethod
    def trim_image(img, spatial_factor, block_size, verbosity=0):
        if verbosity > 0:
            print(Utilities.whoami())
            argdict = locals().copy()
            for k in argdict.keys():
                val = argdict[k]
                if not Utilities.is_iterable(val):
                    print("  {0}: {1}".format(k, argdict[k]))
                else:
                    print("  {0} is iterable".format(k))
        if verbosity > 0:
            print("orig dims: {0}, {1}".format(img.shape[0], img.shape[1]))
        if verbosity > 2:
            import pdb
            pdb.set_trace()
        ht_mod = img.shape[0] % (spatial_factor * block_size)
        wd_mod = img.shape[1] % (spatial_factor * block_size)
        adj_wd = img.shape[0] - ht_mod
        adj_ht = img.shape[1] - wd_mod
        if verbosity > 0:
            print("wd_mod= {0}, ht_mod= {1}\nadj_wd= {2}, adj_ht= {3}".format(wd_mod, ht_mod,
                                                                              adj_wd, adj_ht))
        res = img[0:adj_wd, 0:adj_ht].copy()
        if verbosity > 0:
            print("trimmed dims: {0}, {1}".format(res.shape[0], res.shape[1]))
        return res

    @staticmethod
    def mse(img1, img2):
        res = np.finfo('float').max
        try:
            diff = img1 - img2
            res = np.mean(diff * diff)
        except Exception as e:
            msg = Utilities.last_exception_info()
            warnings.warn(msg)
        return res

    @staticmethod
    def rmse(img1, img2):
        return np.sqrt(ImageUtils.mse(img1, img2))

    @staticmethod
    def center(img, about=0):
        res = img
        mn = None
        try:
            mn = np.mean(img)
            res = img - mn + about
        except Exception as e:
            msg = Utilities.last_exception_info()
            warnings.warn(msg)
        return res, mn


class Compressor(object):
    def __init__(self):
        pass

    @staticmethod
    def find_best_params(img, dimg, rx, ry,
                         block_size, step_size,
                         spatial_factor,
                         intensity_shrinkage,
                         max_x_offset, max_y_offset,
                         err_func=ImageUtils.rmse,
                         verbosity=0):
        if verbosity > 0:
            print("<{0}>".format(ry),end='')
        if verbosity > 1:
            argdict = locals().copy()
            for k in argdict.keys():
                val = argdict[k]
                if not Utilities.is_iterable(val):
                    print("  {0}: {1}".format(k, argdict[k]))
                else:
                    print("  {0} is iterable".format(k))
        if verbosity > 1:
            print("rx= {0}, ry= {1}".format(rx, ry))
        if rx > 0:
            pass
        left = max(int(rx / 2) - max_x_offset, 0)
        right = min(int(rx / 2) + max_x_offset, dimg.shape[1] - block_size )

        up = max(int(ry / 2) - max_y_offset, 0)
        down = min(int(ry / 2) + max_y_offset, dimg.shape[0] - block_size )
        if (left >= right) or (up >= down):
            import pdb
            pdb.set_trace()
        if verbosity > 0:
            pass

        rblock = img[ry:ry + block_size, rx:rx + block_size]
        rmean = np.mean(rblock)

        best_err = np.finfo('float').max
        tries = 0
        best_x = rx
        best_y = ry
        best_mean_adj = 0
        for dx in range(left, right, step_size):
            for dy in range(up, down, step_size):
                temp = dimg[dy: dy + block_size, dx:dx + block_size] * intensity_shrinkage
                if (temp.shape[0] != rblock.shape[0]) or (temp.shape[1] != rblock.shape[1]):
                    msg = Utilities.last_exception_info()
                    warnings.warn(msg)

                dmean = np.mean(temp)
                mean_add = rmean - dmean
                dblock = temp + mean_add
                dblock = np.clip(dblock, 0, 255)
                newmean = np.mean(dblock)
                if (dblock.shape[0] != rblock.shape[0]) or (dblock.shape[1] != rblock.shape[1]):
                    msg = "range and domain have different shapes"
                    msg += "rx={0}, ry={1}".format(rx, ry)
                    raise RuntimeError(msg)
                err = np.finfo('float').max
                try:
                    if (dblock.shape[0] != rblock.shape[0]) or (dblock.shape[1] != rblock.shape[1]):
                        msg = Utilities.last_exception_info()
                        warnings.warn(msg)
                    err = err_func(rblock, dblock)
                except Exception as e:
                    emsg = Utilities.last_exception_info()
                    print(emsg)
                    raise RuntimeError(emsg)
                tries += 1
                if err < best_err:
                    best_x = dx
                    best_y = dy
                    best_mean_add = mean_add
                    best_err = err
        if tries == 0:
            msg = "tries==0, rx={0}, ry= {1}".format(rx, ry)
            raise RuntimeError(msg)
        if best_x > dimg.shape[1] - block_size or best_y > dimg.shape[0] - block_size:
            msg = "codes out of range, x= {0}, y={1}".format(best_x, best_y)
            msg += "image wd= {0}, ht= {1}".format(dimg.shape[1], dimg.shape[0])
            print(msg)
            raise RuntimeError(msg)
        return (best_x, best_y, best_mean_add, rx, ry, best_err, tries)

    @staticmethod
    def compress_image(oimg, block_size=4, step_size=2,
                       spatial_factor=2,
                       intensity_shrinkage=0.75,
                       max_x_offset=None,
                       max_y_offset=None,
                       err_func=ImageUtils.mse,
                       verbosity=0):
        if verbosity > 0:
            print(Utilities.whoami())
            argdict = locals().copy()
            for k in argdict.keys():
                val = argdict[k]
                if not Utilities.is_iterable(val):
                    print("  {0}: {1}".format(k, argdict[k]))
                else:
                    print("  {0} is iterable".format(k))
        if verbosity > 0:
            print("orig dims: {0}, {1}".format(oimg.shape[0], oimg.shape[1]))
        cimg = ImageUtils.trim_image(oimg, spatial_factor=spatial_factor, block_size=block_size, verbosity=verbosity)
        if verbosity > 0:
            print("trimmed dims: {0}, {1}".format(cimg.shape[0], cimg.shape[1]))
        if max_x_offset is None:
            max_x_offset = cimg.shape[1] - block_size
        if max_y_offset is None:
            max_y_offset = cimg.shape[0] - block_size
        dimg = ImageUtils.spatial_shrink(cimg, spatial_factor=spatial_factor)
        print("dimg_wd = {0}, dimg_ht = {1}".format(dimg.shape[0], dimg.shape[1]))
        FCode = namedtuple("FCode", ["dx", "dy", "mean_add", "rx", "ry", "err"])
        codes = []
        for rx in range(0, cimg.shape[1], block_size):
            if verbosity > 0:
                print("rx={0}".format(rx),end='')
            for ry in range(0, cimg.shape[0], block_size):

                parts = Compressor.find_best_params(cimg, dimg, rx, ry,
                                                    block_size=block_size,
                                                    step_size=step_size,
                                                    spatial_factor=spatial_factor,
                                                    intensity_shrinkage=intensity_shrinkage,
                                                    max_x_offset=max_x_offset, max_y_offset=max_y_offset,
                                                    err_func=err_func, verbosity=verbosity)
                dx, dy, mean_add, x, y, err, tries = parts
                code = FCode(dx, dy, mean_add, x, y, err)
                codes.append(code)
            print("--")
        params = OrderedDict()
        params['img_ht'] = cimg.shape[0]
        params['img_wd'] = cimg.shape[1]
        params['block_size'] = block_size
        params['step_size'] = step_size
        params['spatial_factor'] = spatial_factor
        params['intensity_shrinkage'] = intensity_shrinkage
        params['codes'] = codes
        return params


class Decompressor(object):
    def __init__(self):
        pass

    @staticmethod
    def decompress(params,
                   scale_factor=1,
                   iterations=20,
                   initial_image=None,
                   verbosity=0):
        if verbosity > 0:
            print(Utilities.whoami())
            argdict = locals().copy()
            for k in argdict.keys():
                val = argdict[k]
                if not Utilities.is_iterable(val):
                    print("  {0}: {1}".format(k, argdict[k]))
                else:
                    print("  {0} is iterable".format(k))

        if not isinstance(params, dict):
            raise ValueError("input compressed shoule be dict, found {0}".format(type(params)))

        errmsg = ''
        for k in ['img_wd', 'img_ht', 'block_size', 'intensity_shrinkage', 'spatial_factor', 'codes']:
            if k not in params.keys():
                errmsg += "{0} not in compressed.keys()".format(k)
        if errmsg != '':
            raise RuntimeError(errmsg)

        img_wd = params['img_wd'] * scale_factor
        img_ht = params['img_ht'] * scale_factor
        if initial_image is None:
            initial_image = np.zeros([img_ht, img_wd])

        rimg = initial_image.copy()

        block_size = params['block_size'] * scale_factor
        for it in range(iterations):
            ci = 0
            dimg = ImageUtils.spatial_shrink(rimg, spatial_factor=params['spatial_factor'])
            for rx in range(0, img_wd, block_size):
                for ry in range(0, img_ht, block_size):
                    code = params['codes'][ci]
                    temp = dimg[code.dy:code.dy+block_size, code.dx:code.dx+block_size] * params['intensity_shrinkage']
                    if temp.shape[0] != block_size or temp.shape[1] != block_size:
                        msg = "invalid block size"
                    temp = temp + code.mean_add
                    domain = np.clip(temp, 0, 255)
                    rimg[ry:ry+block_size, rx:rx+block_size] = domain
                    ci += 1
        return rimg.copy()


if __name__ == '__main__':
    colimg = mpimg.imread("fopu-staircase.JPG")
    big_img = ImageUtils.rgb2gray(colimg)
    print(big_img.shape)
    small_img = ImageUtils.spatial_shrink(big_img, spatial_factor=32)
    print(small_img.shape)

    # params
    spatial_factor = 2
    block_size = 2
    step_size = 2
    verbosity = 1
    intensity_shrinkage = 0.7

    oimg = ImageUtils.trim_image(small_img, spatial_factor=spatial_factor,
                                 block_size=block_size,
                                 verbosity=verbosity)

    # show image
    plt.imshow(oimg, cmap=plt.get_cmap('gray'))  #, vmin=0, vmax=1)
    plt.show()

    # compress
    Comp = Compressor()
    start = Utilities.now()
    print(start)
    params = Comp.compress_image(oimg, block_size=block_size, spatial_factor=spatial_factor,
                                       intensity_shrinkage=intensity_shrinkage,
                                       err_func=ImageUtils.rmse,
                                       max_x_offset=None, max_y_offset=None,
                                       verbosity=1)
    end = Utilities.now()
    print(end)
    cdf = pd.DataFrame(params['codes'])
    print("rmse= {0}".format(np.sqrt(cdf['err'].mean())))
    # decompress
    Decomp = Decompressor()
    dimages = []
    errors = []
    changes = []
    next_image = Decompressor.decompress(params, iterations=1)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 6),
                            subplot_kw={'xticks': [], 'yticks': []})
    axs[0].imshow(oimg, cmap=plt.get_cmap('gray'))  # , vmin=0, vmax=1)
    axs[1].imshow(next_image, cmap=plt.get_cmap('gray'))  # , vmin=0, vmax=1)
    plt.show()
    dimages.append(next_image)
    for it in range(20):
        last_image = next_image.copy()
        next_image = Decompressor.decompress(params, iterations=1, initial_image=last_image)
        error = ImageUtils.rmse(oimg, next_image)
        errors.append(error)
        dimages.append(next_image)
        change = ImageUtils.mse(last_image, next_image)
        changes.append(change)
        print("change in images {0}".format(change))
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 6),
                                subplot_kw={'xticks': [], 'yticks': []})
        axs[0].imshow(oimg, cmap=plt.get_cmap('gray'))  #, vmin=0, vmax=1)
        axs[1].imshow(next_image, cmap=plt.get_cmap('gray'))  #, vmin=0, vmax=1)
        plt.show()
        print("{0}".format(it))
    print("done")