from __future__ import print_function

import pickle

import matplotlib.pyplot as plt
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
import warnings as warn
import os
import csv
import matplotlib.image as mpimg
import tempfile
import urllib
from skimage.transform import resize
from PIL import Image
import random
import scipy.io as scio
import shutil
import inspect
import logging

import datetime
import scipy.stats as st

# from utils.visualization import generate_montage

__author__ = 'Joseph Robinson'


def landmarks2heatmaps(landmarks, filter, imsize=(80, 80), sigma=1.0):
    """ Draw"""
    landmarks = landmarks*imsize[0] if np.all(landmarks <= 1.0) else landmarks

    heatmaps = []
    for k, lmark in enumerate(landmarks):
        heatmap = np.zeros((imsize[0], imsize[1]))
        if np.all(lmark > 0):
            heatmap[int(np.round(lmark[1])) - 1, int(np.round(lmark[0])) - 1] = 1.0

        heatmaps.append(filter(heatmap, sigma))


    return np.array(heatmaps)


def coordinates2mask(coords, sh=(80, 80)):
    mask = np.zeros(sh)
    for pt in coords:
        mask[int(pt[0]), int(pt[1])] = 1
    return mask


def mask2coordinates(mask):
    indices = np.where(mask != [0])
    landmarks = [[ind[0], ind[1]] for ind in zip(indices[0], indices[1])]

    return np.array(landmarks)


def gkern(kernlen=5, nsig=1):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, m=1, n=1):
        self.val = val
        self.sum += val * n
        self.count += n * m
        self.avg = self.sum / self.count



def get_time_minutes():
    return datetime.datetime.now().hour * 60 + datetime.datetime.now().minute + datetime.datetime.now().second / 60

def gray2rgb(im):
    """converting a x by y array of floats into a x by y by 3 array of 8-bit ints"""
    return np.dstack([im.astype(np.uint8)] * 3)


def autolog(message):
    "Automatically log the current function details."
    func = inspect.currentframe().f_back.f_code
    logging.debug("%s: %s in %s:%i" % (
        message,
        func.co_name,
        func.co_filename,
        func.co_firstlineno
    ))



def list(imdir):
    """return list of images with absolute path in a directory"""
    return [os.path.abspath(os.path.join(imdir, item)) for item in os.listdir(imdir) if
            (is_img(item) and not is_hidden_file(item))]


def save_image_pil(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def savelist(imdir, outfile):
    """Write out all images in a directory to a provided file with each line containing absolute path to image"""
    return writelist(list(imdir), outfile)


def read(img_file):
    """Read image from file
    :param img_file: filepath of image to open and return
    :type img_file: string
    """
    if not is_file(img_file):
        return None
    return mpimg.imread(img_file)


def resize(img):
    """
    :param img:
    :return:
    """
    img = np.true_divide(img - np.amin(img), np.amax(img) - np.amin(img))

    return img



def rgb2gray(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return 0.299 * R + 0.587 * G + 0.114 * B


def rgb_to_csv(img, ofile):
    """Write 3-channel numpy matrix (i.e., RGB img) to csv, with each column
    containing a single channel and each channel vectorized with elements
    ordered down columns"""

    width, height, channel = img.shape
    with open(ofile, 'w+') as f:
        f.write('R,G,B\n')
        # read the details of each pixel and write them to the file
        for x in range(width):
            for y in range(height):
                r = img[x, y][0]
                g = img[x, y][1]
                b = img[x, y][2]
                f.write('{0},{1},{2}\n'.format(r, g, b))


def temp(ext='jpg'):
    """Create a temporary image with the given extension"""
    if ext[0] == '.':
        ext = ext[1:]
    return tempfile.mktemp() + '.' + ext


def temp_png():
    """Create a temporay PNG file"""
    return temp('png')





def normalize(img):
    """Normalize image to have no negative numbers"""
    imin = np.min(img)
    imax = np.max(img)

    if (imax - imin):
        return (img - imin) / (imax - imin)
    else:
        img


def get_image_list(in_file):
    f_images = []
    with open(in_file) as f:
        contents = csv.reader(f, delimiter=',')
        for x, row in enumerate(contents):
            if x == 0:
                continue
            val = row[0].strip()
            f_images.append(val)
    return f_images


def csv_list(imdir):
    """Return a list of absolute paths of *.csv files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir) if is_csv(item)]


def dir_list(indir):
    """return list of directories in a directory"""
    return [os.path.abspath(os.path.join(indir, item)) for item in os.listdir(indir) if
            (os.path.isdir(os.path.join(indir, item)) and not is_hidden_file(item))]

def file_name(filename):
    """Return c.ext for filename /a/b/c.ext"""
    return os.path.split(filename)[1]


def file_base(filename):
    """Return c for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    (base, ext) = os.path.splitext(tail)
    return base


def parent_dir_name(dirpath):
    """Return b from path /a/b/c.ext or  /a/b/"""
    return str.split(parent_dir(dirpath + "/"), '/')[-1]


def file_ext(filename):
    """Given filename /a/b/c.ext return .ext"""
    (head, tail) = os.path.split(filename)
    try:
        parts = str.rsplit(tail, '.', 2)
        if len(parts) == 3:
            ext = '.%s.%s' % (parts[1], parts[2])  # # tar.gz
        else:
            ext = '.' + parts[1]
    except:
        ext = None

    return ext


def parent_dir(filename):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return head


def pklist(imdir):
    """Return a list of absolute paths of *.pk files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir) if is_pickle(os.path.join(imdir, item))]


def file_tail(filename):
    """Return c.ext for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return tail


def is_img(path):
    """Is object an image with a known extension ['.jpg','.jpeg','.png','.tif','.tiff','.pgm','.ppm','.gif','.bmp']?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pgm', '.ppm', '.gif', '.bmp']


def is_pickle(filename):
    """Is the file a pickle archive file"""
    return is_file(filename) and os.path.exists(filename) and file_ext(filename).lower() in ['.pk', '.pkl']


def is_text_file(path):
    """Is the given file a text file?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.txt'] and (filename[0] != '.')


def is_video(path):
    """Is a file a video with a known video extension ['.avi','.mp4','.mov','.wmv','.mpg']?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.avi', '.mp4', '.mov', '.wmv', 'mpg']

def is_npy(path):
    """Is a file a npy file extension?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.npy', '.NPY']


def is_csv(path):
    """Is a file a CSV file extension?"""
    (filename, ext) = os.path.splitext(path)
    return ext.lower() in ['.csv', '.CSV']


def is_file(path):
    """Wrapper for os.path.is_file"""
    return os.path.isfile(str(path))


def is_dir(path):
    """Wrapper for os.path.isdir"""
    return os.path.isdir(path)


def is_hidden_file(filename):
    """Does the filename start with a period?"""
    return filename[0] == '.'


def load_mat(matfile):
    return scio.loadmat(matfile)


def load_pickle(picklefile):
    with open(picklefile, 'rb') as fp:
        return pickle.load(fp)


def readcsv(infile, separator=','):
    """Read a csv file into a list of lists"""
    with open(infile, 'r') as f:
        list_of_rows = [[x.strip() for x in r.split(separator)] for r in f.readlines()]
    return list_of_rows


def readlist(infile):
    """Read each row of file as an element of the list"""
    with open(infile, 'r') as f:
        list_of_rows = [r for r in f.readlines()]
    return list_of_rows


def read_mat(txtfile, delimiter=' '):
    """Whitespace separated values defining columns, lines define rows.  Return numpy array"""
    with open(txtfile, 'rb') as csvfile:
        M = [np.float32(row.split(delimiter)) for row in csvfile]
    return np.array(M)


def readtxt(ifile):
    """ Simple function to read text file and remove clean ends of spaces and \n"""
    with open(ifile, 'r') as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def sys_home():
    """
    :return: Home directory (platform agnostic)
    """
    return os.path.expanduser("~")


def mkdir(output):
    """
    Make directory if does not already exist.
    :param output:
    :return:    True if no directory exists, and 'output' was made; else, False.
    """
    if not os.path.exists(output):
        os.makedirs(output)
        return True
    return False


def mkdirs(paths):
    """
    Make directories that do not already exist.
    :param paths:
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def delete_dirs(paths):
    """
    deletes directories recursively
    :type paths: object
    :param paths:
    :return:
    """

    if len(paths) > 0 and not isinstance(paths, str):
        for path in paths:
            shutil.rmtree(path)
    else:
        shutil.rmtree(paths)


def filepath(filename):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return head


def newpath(filename, newdir):
    """Return /a/b for filename /a/b/c.ext"""
    (head, tail) = os.path.split(filename)
    return os.path.join(newdir, tail)


def videolist(videodir):
    """return list of images with absolute path in a directory"""
    return [os.path.abspath(os.path.join(videodir, item)) for item in os.listdir(videodir) if
            (is_video(item) and not is_hidden_file(item))]


def rand(start, end, num):
    """
    Function to generate and append them
    :param start:   starting range
    :param end:     ending range
    :param num:     number of elements needs to be appended
    :return:
    """
    res = []

    for j in range(num):
        res.append(random.randint(start, end))

    return res


def writecsv(list_of_tuples, outfile, mode='w', separator=','):
    """Write list of tuples to output csv file with each list element on a row and tuple elements separated by comma"""
    list_of_tuples = list_of_tuples if not is_numpy(list_of_tuples) else list_of_tuples.tolist()
    with open(outfile, mode) as f:
        for u in list_of_tuples:
            n = len(u)
            for (k, v) in enumerate(u):
                if (k + 1) < n:
                    f.write(str(v) + separator)
                else:
                    f.write(str(v) + '\n')
    return (outfile)


def writelist(mylist, outfile, mode='w'):
    """Write list of strings to an output file with each row an element of the list"""
    with open(outfile, mode) as f:
        for s in mylist:
            f.write(str(s) + '\n')
    return (outfile)


def check_paths(*paths):
    """
    Function that checks variable number of files (i.e., unordered arguments, *paths). If any of the files do not
    exist '
    then function fails (i.e., no info about failed indices, but just pass (True) or fail (False))
    :param paths:   unordered args, each pointing to file.
    :return:
    """
    do_exist = True
    for x, path in enumerate(paths):
        if not os.path.isfile(path):
            warn.warn(str(x) + ") File not found: " + path)
            do_exist = False

    return do_exist


def is_linux():
    """is the current platform Linux?"""
    (sysname, nodename, release, version, machine) = os.uname()
    return sysname == 'Linux'


def is_macosx():
    """Is the current platform MacOSX?"""
    (sysname, nodename, release, version, machine) = os.uname()
    return sysname == 'Darwin'


def is_number(obj):
    """Is a python object a number?"""
    try:
        complex(obj)  # for int, long, float and complex
    except ValueError:
        return False

    return True


def is_numpy(obj):
    """Is a python object a numpy array?"""
    return 'numpy' in str(type(obj))


def parse(cfile=None):
    """
    Instantiates parser for INI (config) file
    :param cfile: absolute filepath to config INI file
    :return: ConfigParser object with configurations loaded
    """
    if not cfile:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cfile = os.path.join(dir_path, 'my_bb_configs.ini')

    print('Loading configs: ' + cfile)
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(cfile)

    return parser


# end parse()


def print_configs(opts, header=''):
    """
    Simple function that prints configurations
    :param opts: configs of certain type (named tuple)
    :param header: Title (reference) to display above printout
    """
    if header:
        print('\n###############################################################\n')
        print('\t########\t {} \t########\n'.format(header))
        print('###############################################################\n')

    for field in opts._fields:
        if len(field) < 8:
            print('\t{}\t\t\t:\t{}\n'.format(field, getattr(opts, field)))
        else:
            print('\t{}\t\t:\t{}\n'.format(field, getattr(opts, field)))


def show(img_display, img, lmarks, frontal_raw, face_proj, background_proj, temp_proj2_out_2, sym_weight):
    plt.ion()
    plt.show()
    plt.subplot(221)
    plt.title('Query Image')
    plt.imshow(img_display[:, :, ::-1])
    plt.axis('off')

    plt.subplot(222)
    plt.title('Landmarks Detected')
    plt.imshow(img[:, :, ::-1])
    plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1], c='red', marker='.', s=100, alpha=0.5)
    plt.axis('off')
    plt.subplot(223)
    plt.title('Rendering')

    plt.imshow(frontal_raw[:, :, ::-1])
    plt.axis('off')

    plt.subplot(224)
    if sym_weight is None:
        plt.title('Face Mesh Projected')
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        face_proj = np.transpose(face_proj)
        plt.plot(face_proj[1:-1:100, 0], face_proj[1:-1:100, 1], 'b.')
        background_proj = np.transpose(background_proj)
        temp_proj2_out_2 = temp_proj2_out_2.T
        plt.plot(background_proj[1:-1:100, 0], background_proj[1:-1:100, 1], 'r.')
        plt.plot(temp_proj2_out_2[1:-1:100, 0], temp_proj2_out_2[1:-1:100, 1], 'm.')
    else:
        plt.title('Face Symmetry')
        plt.imshow(sym_weight)
        plt.axis('off')
        plt.colorbar()

    plt.draw()
    plt.pause(0.001)
    __ = input("Press [enter] to continue.")
    plt.clf()


def txtlist(imdir):
    """Return a list of absolute paths of *.txt files in current directory"""
    return [os.path.join(imdir, item) for item in os.listdir(imdir) if is_text_file(item) and not is_hidden_file(item)]

