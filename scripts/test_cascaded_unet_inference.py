# Python 2.7.6 test script to test the following Caffe model on our CT liver images
# Translated from cascaded_unet_inference.ipynb to test CD'or 3.0 liver segmentation
# ref https://modelzoo.co/model/cascaded-fully-convolutional-networks-for-biomedical
# ref https://github.com/IBBM/Cascaded-FCN
import os
import sys
import wget  #keesh added to replace !wget
import argparse  #keesh added to add options

import caffe  #which version to build?

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as imsave
from IPython import display

import scipy
import scipy.misc

import dicom
import natsort
import glob

#globals for now
STEP1_DEPLOY_PROTOTXT = "../models/cascadedfcn/step1/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "../models/cascadedfcn/step1/step1_weights.caffemodel"
IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

def main(inpArgs):
    try:
        # Get model weights (step1 models)
        # ref https://stackoverflow.com/questions/24346872/python-equivalent-of-a-given-wget-command
        output_file1 = "../models/cascadedfcn/step1/step1_weights.caffemodel"
        if os.path.isfile(output_file1) == False:
            url1 = "https://www.dropbox.com/s/aoykiiuu669igxa/step1_weights.caffemodel?dl=1"
            print("Getting model 1 weights: " + output_file1)
            wget.download(url1, out=output_file1)

        print("caffe__file: " + caffe.__file__)
        # Use CPU for inference
        # May get 1394 error -- hopefully not serious
        # also Check failed: status == CUDNN_STATUS_SUCCESS (1 vs. 0)  CUDNN_STATUS_NOT_INITIALIZED
        caffe.set_mode_cpu()
        # Use GPU for inference -- does this ever work any any VM?
        #caffe.set_mode_gpu()

        plt.set_cmap('gray')

        perform_inference(os.path.abspath(inpArgs.input_dicom_dir), os.path.abspath(inpArgs.output_results_dir), int(inpArgs.slice_skip),
                          inpArgs.apply_user_wl, inpArgs.apply_hist_eq)

        sys.exit(0)
    except IOError as ioex:
        print("There was an IO error: " + str(ioex.strerror))
        sys.exit(1)

    except Exception as e:
        print("There was an unexpected error: " + str(e))
        sys.exit(1)


""" Image I/O  """
def read_dicom_series(directory, filepattern = "image_*"):
    """ Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""
    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    print('\tRead Dicom dir: ' + str(directory))
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    print('\tLength dicom series: ' + str(len(lstFilesDCM)))
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    first_time = True
    wc = ww = 0
    b = 0
    m = 1
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        if first_time:
            try:
                wc = int(ds[0x0028, 0x1050].value)   #0028,1050  Window Center: 40
                ww = int(ds[0x0028, 0x1051].value)   #0028,1051  Window Width: 400
                b = float(ds[0x0028, 0x1052].value)  #0028,1052  Rescale Intercept: -1024
                m = float(ds[0x0028, 0x1053].value)  #0028,1053  Rescale Slope: 1
                first_time = False
            except:
                wc = ww = 0  #not needed but clearer
                b = 0
                m = 1
    return ArrayDicom, len(lstFilesDCM), wc, ww, b, m


""" Image Stats / Display"""
def stat(array):
    #may need str casts?
    print('min: ' + np.min(array) + ' max: ' + np.max(array) + ' median: ' + np.median(array) + ' avg: ' + np.mean(array))



""" Image Preprocessing """
def to_scale(img, shape=None):
    height, width = shape
    if img.dtype == SEG_DTYPE:
        # This function is only available if Python Imaging Library (PIL) is installed.
        # 888 Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
        return scipy.misc.imresize(img,(height,width),interp="nearest").astype(SEG_DTYPE)
    elif img.dtype == IMG_DTYPE:
        max_ = np.max(img)
        factor = 255.0/max_ if max_ != 0 else 1
        return (scipy.misc.imresize(img,(height,width),interp="nearest")/factor).astype(IMG_DTYPE)
    else:
        raise TypeError('Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')

def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)

def normalize_image_using_rescale_slope_intercept(img, m, b):
    """ Normalize image values to y = mx + b """
    return ((m * img) + b)

def histeq_processor(img):
    """Histogram equalization"""
    nbr_bins=256
    #get image histogram
    imhist,bins = np.histogram(img.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    #use linear interpolation of cdf to find new pixel values
    original_shape = img.shape
    img = np.interp(img.flatten(),bins[:-1],cdf)
    img=img/255.0
    return img.reshape(original_shape)

def step1_preprocess_img_slice(img_slc, slice, wc, ww, b, m, apply_user_wl, apply_hist_eq, results_dir):
    """
    Preprocesses the image 3d volumes by performing the following :
    1- Rotate the input volume so the the liver is on the left, spine is at the bottom of the image
    2- Set pixels with hounsfield value great than 1200, to zero.
    3- Clip all hounsfield values to the range [-100, 400]
    4- Normalize values to [0, 1]
    5- Rescale img and label slices to 388x388
    6- Pad img slices with 92 pixels on all sides (so total shape is 572x572)

    Args:
        img_slc: raw image slice
    Return:
        Preprocessed image slice
    """
    img_slc   = img_slc.astype(IMG_DTYPE)

    #apply m and b
    img_slc = normalize_image_using_rescale_slope_intercept(img_slc, m, b)
    print("m= " + str(m))
    print("b= " + str(b))

    img_slc[img_slc>1200] = 0

    thresh_lo = -100
    thresh_hi = 400
    if apply_user_wl:
        if wc != 0 and ww != 0:
            thresh_lo = wc - (ww / 2)
            thresh_hi = wc + (ww / 2)
    print("HU thresh low= " + str(thresh_lo))
    print("HU thresh high= " + str(thresh_hi))

    img_slc   = np.clip(img_slc, thresh_lo, thresh_hi)

    # save HU image to disk
    slice_lbl = "slice" + str(slice)
    fname = results_dir + os.path.sep + 'preproc1hu_' + slice_lbl + '.png'
    imsave.imsave(fname, img_slc)

    img_slc   = normalize_image(img_slc)
    img_slc   = to_scale(img_slc, (388,388))
    img_slc   = np.pad(img_slc,((92,92),(92,92)),mode='reflect')

    if apply_hist_eq:
        img_slc = histeq_processor(img_slc)

    return img_slc


def perform_inference(input_dir, results_dir, mod_slices, apply_user_wl, apply_hist_eq):
    """ Read Test Data """
    img, num_images, wc, ww, b, m = read_dicom_series(input_dir + os.path.sep, filepattern="*.dcm")

    # process an image every every x slices
    if os.path.isdir(results_dir) == False:
        os.mkdir(results_dir)
    # Load network
    net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)
    print("step 1 net constructed")
    for slice in range(0, num_images, mod_slices):
        slice_lbl = "slice" + str(slice)
        fname = results_dir + os.path.sep + 'orig_' + slice_lbl + '.png'
        imsave.imsave(fname, img[...,slice])

        # Prepare a test slice
        # May have to flip left to right (and change assumptions like HU thresholds)
        img_p = step1_preprocess_img_slice(img[...,slice], slice, wc, ww, b, m, apply_user_wl, apply_hist_eq, results_dir)

        fname = results_dir + os.path.sep + 'preproc1_' + slice_lbl + '.png'
        imsave.imsave(fname, img_p)

        """ Perform Inference """
        # Predict
        net1.blobs['data'].data[0,0,...] = img_p
        print("fed slice: " + slice_lbl)

        pred = net1.forward()['prob'][0,1] > 0.5
        print("pred1 mask: "+ slice_lbl)

        # May wish to convert to byte data type
        print("pred shape, type:" + pred.shape + "," + type(pred))

        fname = results_dir + os.path.sep + 'pred1_mask_' + slice_lbl + '.png'
        # Visualize results
        imsave.imsave(fname, (pred>0.5))

    # Free up memory of step1 network
    del net1  #needed ?


if __name__ == '__main__':
    '''
    This script runs step 1 of the Cascaded-FCN using its CT liver model on a test dicom dir.
    '''
    parser = argparse.ArgumentParser(description='step 1 of Cascaded-FCN test script')
    parser.add_argument("-i", dest="input_dicom_dir", help="The input dicom directory to read test images from")
    parser.add_argument("-o", dest="output_results_dir", help="The output directory to write results to")
    parser.add_argument("-s", dest="slice_skip", help="How many slices to skip via mod test")
    parser.add_argument("-w", "--apply_user_wl", dest="apply_user_wl", help="Whether to apply user-defined W/L in pre-processing")
    parser.add_argument("-h", "--apply_hist_eq", dest="apply_hist_eq", help="Whether to apply histogram equalization in pre-processing")
    if len(sys.argv) < 6:
        print("python test_cascaded_unet_inference.py -i <input_dcm_dir> -o <output_results_dir> -s 20 --apply_user_wl True --apply_hist_eq False")
        sys.exit(1)
    inpArgs = parser.parse_args()
    main(inpArgs)

