import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


ALLTYPES = ['b', 'c', 'f', 'f2', 'f3']
MAX_TYPE_COUNT = {'b':3, 'c':4, 'f':7, 'f2':3, 'f3':5}

def load_vmat_defaults(files, args):
    if not hasattr(args, 'matdefaults'):
        args.matdefaults = {}
        for item in files:
            fname, matname = item
            subid, classid, boolgt, boolmask, colgt, colmask, fgt, fmask, \
                f2gt, f2mask, f3gt, f3mask = load_vmat_gt(fname)
            vals = {}
            vals['subid'] = subid
            vals['classid'] = classid
            vals['b'] = [boolgt, boolmask]
            vals['c'] = [colgt, colmask]
            vals['f'] = [fgt, fmask]
            vals['f2'] = [f2gt, f2mask]
            vals['f3'] = [f3gt, f3mask]
            args.matdefaults[matname] = vals

def load_vmat_gt(fname):
    lines = open(fname).read().splitlines()
    subid = int(lines[0])
    classid = int(lines[1])
    boolgt = np.zeros((MAX_TYPE_COUNT['b'], 1), dtype=int)
    boolmask = np.zeros((MAX_TYPE_COUNT['b'], 1), dtype=int)
    colgt = np.zeros((MAX_TYPE_COUNT['c'], 3))
    colmask = np.zeros((MAX_TYPE_COUNT['c'], 3), dtype=int)
    fgt = np.zeros((MAX_TYPE_COUNT['f'], 1))
    fmask = np.zeros((MAX_TYPE_COUNT['f'], 1), dtype=int)
    f2gt = np.zeros((MAX_TYPE_COUNT['f2'], 2))
    f2mask = np.zeros((MAX_TYPE_COUNT['f2'], 2), dtype=int)
    f3gt = np.zeros((MAX_TYPE_COUNT['f3'], 3))
    f3mask = np.zeros((MAX_TYPE_COUNT['f3'], 3), dtype=int)
    cur = 1
    for i in range(MAX_TYPE_COUNT['b']):
        cur += 1
        val, vrange = lines[cur].split(',')
        boolgt[i] = int(val)
    for i in range(MAX_TYPE_COUNT['b']):
        cur += 1
        boolmask[i] = int(lines[cur])
    for i in range(MAX_TYPE_COUNT['c']):
        cur += 1
        val, vrange = lines[cur].split(',')
        xmin, xmax = [float(k) for k in vrange.split(" ")]
        colgt[i] = [float(k) for k in val.split(" ")]
    for i in range(MAX_TYPE_COUNT['c']):
        cur += 1
        colmask[i] = [int(k) for k in lines[cur].split(" ")]
    for i in range(MAX_TYPE_COUNT['f']):
        cur += 1
        val, vrange = lines[cur].split(",")
        xmin, xmax = [float(k) for k in vrange.split(" ")]
        fgt[i] = float(val)
        if xmax > 1:
            fgt[i] = fgt[i]/xmax
    for i in range(MAX_TYPE_COUNT['f']):
        cur += 1
        fmask[i] = [int(k) for k in lines[cur].split(" ")]
    for i in range(MAX_TYPE_COUNT['f2']):
        cur += 1
        val, vrange = lines[cur].split(",")
        xmin, xmax = [float(k) for k in vrange.split(" ")]
        f2gt[i] = [float(k) for k in val.split(" ")]
        if xmax > 1:
            f2gt[i] = [float(k)/xmax for k in val.split(" ")]
    for i in range(MAX_TYPE_COUNT['f2']):
        cur += 1
        f2mask[i] = [int(k) for k in lines[cur].split(" ")]
    for i in range(MAX_TYPE_COUNT['f3']):
        cur += 1
        val, vrange = lines[cur].split(",")
        xmin, xmax = [float(k) for k in vrange.split(" ")]
        f3gt[i] = [float(k) for k in val.split(" ")]
        if xmax > 1:
            f3gt[i] = [float(k)/xmax for k in val.split(" ")]
    for i in range(MAX_TYPE_COUNT['f3']):
        cur += 1
        f3mask[i] = [int(k) for k in lines[cur].split(" ")]
    return np.array([subid]), np.array([classid]), boolgt, boolmask, colgt, colmask, fgt, fmask, \
        f2gt, f2mask, f3gt, f3mask

def load_norm_ranges(fname):
    normdict = {}
    lines = [k.split(',') for k in open(fname).read().splitlines()]
    for line in lines:
        name = line[0]
        normdict[name] = {}
        for param in line[1:]:
            pname, xmin, xmax = param.split(" ")
            normdict[name][pname] = [xmin, xmax]
    return normdict

def load_img(args, path, method=1):
    img = cv2.imread(path, method)
    if method == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def cv_imshow(name, data):
    cv2.imshow(name, data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_img_gt(args, title, data):
    fig = plt.figure()
    n_images = len(data)
    cols = 1
    for i in range(n_images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1)
        plot = plt.imshow(data[i])
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        a.set_title(title[i])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.tight_layout()
    fig.patch.set_alpha(1)
    plt.show(block=True)


def save_img_gt(args, title, data, bid, img_id, epoch):
    fig = plt.figure()
    n_images = len(data)
    cols = 1
    for i in range(n_images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), i + 1)
        plot = plt.imshow(data[i])
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        a.set_title(title[i])
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.tight_layout()
    fig.patch.set_alpha(1)
    plot_path = os.path.join(args.odir, 'images', str(epoch), str(bid))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, title[0].split('/')[-1].replace('.png', str(img_id) + '.png')), facecolor=fig.get_facecolor())
    plt.close()


def load_data_args(args):
    pass
