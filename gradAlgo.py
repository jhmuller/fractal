from collections import namedtuple
#from collections import OrderedDict
import datetime
import numpy as np
import warnings
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from utilities import Utilities

class FractalMaker(object):
    def __init__(self):
        pass

    @staticmethod
    def make_fractal(ifs_df,
                     xoffset=0,
                     yoffset=0,
                     image_ht=320, image_wd=320,
                     n_iterations=3100,
                     verbosity=0):

        ifs_df['cum_p'] = ifs_df['p'].cumsum()
        if verbosity > 0:
            print(ifs_df)
        # initialize image
        yscale = image_ht * 0.85
        xscale = image_wd
        #xoffset = image_wd/2
        #yoffset = 10
        byte_array = [255] * (image_ht * image_wd)
        image = np.array(byte_array)
        image = image.reshape(image_ht, image_wd)
        print("image shape {0} {1}".format(image.shape[0], image.shape[1]))
        x = 0
        y = 0
        kprobs = [0]*ifs_df.shape[0]
        try:
            print(datetime.datetime.now())
            trajectory = []
            written = 0
            for n in range(n_iterations):
                trajectory.append([x,y])
                if (n % 10000) == 0:
                    print("iteration {0}".format(n))
                pk = np.random.random(1)[0]
                k = 0
                while pk > ifs_df["cum_p"][k]:
                    k += 1
                kprobs[k] += 1
                xnew = ifs_df["a"][k]*x +\
                       ifs_df["b"][k]*y +\
                       ifs_df["e"][k]

                ynew = ifs_df["c"][k]*x +\
                       ifs_df["d"][k]*y +\
                       ifs_df["f"][k]

                x = xnew
                y = ynew
                if n > 400:
                    posx = int(np.round(x * xscale + xoffset, 1))
                    posy = int(np.round(y * yscale + yoffset, 1))

                    if posx >= image_wd or posy >= image_ht or\
                       posx < 0 or posy < 0:
                        msg = "position {0}, {1} out of range {2} , {3}".format(posx, posy,
                                                                                image_wd,
                                                                                image_ht)
                        #raise ValueError(msg)
                    else:
                        old = image[posy, posx]
                        image[posy, posx] = 0 #- old
                        written = written + 1

            kprobs = kprobs / np.sum(kprobs)
            print(kprobs)
            print("written= {0}".format(written))
            print(datetime.datetime.now())
        except Exception as e:
            msg = Utilities.last_exception_info()
            warnings.warn(msg)
            raise ValueError(msg)
        return image


if __name__ == "__main__":
    IFS_TUP = namedtuple("IFS_TUP", ["a", "b", "c","d","e","f", "p"])

    ifs_list = list()
    ifs_list.append(IFS_TUP(0,     0,    0,  0.16,  0,    0,  .01))
    ifs_list.append(IFS_TUP(.2,  -.26,  .23, 0.22,  0,  .16,  .07))
    ifs_list.append(IFS_TUP(-.15, .28,  .26, 0.24,  0,   .44, .07))
    ifs_list.append(IFS_TUP(.85,  .04, -.04,  .85,  0,  .166,  .85))
    fern_df = pd.DataFrame(ifs_list)
    print(fern_df)

    ifs_list = list()
    ifs_list.append(IFS_TUP(.5, 0, 0, .5,  0, 0,  0.33))
    ifs_list.append(IFS_TUP(.5, 0, 0, .5,  1, 0,  0.33))
    ifs_list.append(IFS_TUP(.5, 0, 0, .5, .5, .5, 0.34))
    serpinski_df = pd.DataFrame(ifs_list)
    print(serpinski_df)

    Fmaker = FractalMaker()
    ifs_df = serpinski_df
    fimage = Fmaker.make_fractal(ifs_df=ifs_df,
                                 xoffset=0,
                                 yoffset=0,
                                 image_ht=800, image_wd=800,
                                 n_iterations=90000,
                                 verbosity=1)
    fimage = np.flip(fimage, axis=0)
    fig, ax = plt.subplots(nrows=1, ncols=1,
                            figsize=(20,20),
                            #subplot_kw={'xticks': [], 'yticks': []}
                           )
    ax.imshow(fimage, cmap=plt.get_cmap('gray'))  # , vmin=0, vmax=1)
    plt.show()
    print("done")