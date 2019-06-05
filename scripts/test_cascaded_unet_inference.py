# Python 2.7.6 test script to test the following Caffe model on our CT liver images
# Translated from cascaded_unet_inference.ipynb to test CD'or 3.0 liver segmentation
# ref https://modelzoo.co/model/cascaded-fully-convolutional-networks-for-biomedical
# ref https://github.com/IBBM/Cascaded-FCN
import os
import sys
import wget

# keesh added to replace !wget
import argparse  # keesh added to add options

import caffe  # which version to build?

import numpy as np
import scipy
import scipy.misc

import dicom
from dicom.dataset import Dataset
from dicom.dataset import FileDataset
import datetime
import time
import platform

import natsort
import glob

#globals for now
STEP1_DEPLOY_PROTOTXT = "../models/cascadedfcn/step1/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "../models/cascadedfcn/step1/step1_weights.caffemodel"
IMG_DTYPE = np.float
SEG_DTYPE = np.uint8
first_time = True  # move to class when prototype is stable

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
        # caffe.set_mode_cpu()
        # Use GPU for inference -- Need exact CUDA version/driver synched with Caffe!
        caffe.set_mode_gpu()

        test_feature = False
        if inpArgs.test_feature.lower() == "true":  # "true" or "false" as a string
            test_feature = True

        perform_inference(os.path.abspath(inpArgs.input_dicom_dir), os.path.abspath(inpArgs.output_results_dir), test_feature)

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
    Arrayds = [None] * len(lstFilesDCM)

    # loop through all the DICOM files
    b = 0
    m = 1
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        Arrayds[lstFilesDCM.index(filenameDCM)] = ds
        if first_time:
            # DICOM type 1 required
            b = float(ds[0x0028, 0x1052].value)  # 0028,1052  Rescale Intercept: -1024
            m = float(ds[0x0028, 0x1053].value)  # 0028,1053  Rescale Slope: 1
            print("b= " + str(b))
            print("m= " + str(m))

    return ArrayDicom, Arrayds, len(lstFilesDCM), b, m

def write_dicom_mask(img_slice, ds_slice, slice_no, outputdirectory, filepattern = ".dcm"):
    file_meta = Dataset()
    #will need to generate all UID  uniqly see Mayo Image Studio
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'

    series_number = ds_slice[0x0020, 0x0011].value
    base_fname = str(slice_no).zfill(6)
    filename = outputdirectory + os.path.sep + base_fname + "_" + str(series_number) + "_mask1" + filepattern
    ds = FileDataset(filename, {}, file_meta = file_meta, preamble="\0"*128)
    ds.Modality = ds_slice.Modality
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.StudyInstanceUID = '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    ds.SOPClassUID = 'Secondary Capture Image Storage'
    ds.SecondaryCaptureDeviceManufacturer = platform.sys.version

    # These are the necessary imaging components of the FileDataset object.
    (rows, cols) = img_slice.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 7
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\x01\\x01'
    ds.Columns = cols
    ds.Rows = rows
    ds.PixelData = img_slice.tobytes()

    ds.ImplementationVersionName = "pydicom"  #should add version too
    image_type_val = ds_slice[0x0008, 0x0008].value
    image_type_val_str = "\\".join(str(x) for x in image_type_val)
    image_type_val_str2 = image_type_val_str.replace("ORIGINAL", "DERIVED", 1)
    ds.ImageType = image_type_val_str2

    ds.SliceThickness = ds_slice[0x0018, 0x0050].value

    #these tags may be missingg
    try:
        ds.SpacingBetweenSlices = ds_slice[0x0018, 0x0088].value
        ds.SliceLocation = ds_slice[0x0020, 0x1041].value
    except:
        pass

    ds.SeriesNumber = series_number
    ds.InstanceNumber = ds_slice[0x0020, 0x0013].value

    ds.ImagePositionPatient = ds_slice[0x0020, 0x0032].value # 0020,0032  Image Position (Patient): 0\0\0
    ds.ImageOrientationPatient = ds_slice[0x0020, 0x0037].value # 0020,0037  Image Orientation (Patient): 1\0\0\0\1\0

    ds.PixelSpacing = ds_slice[0x0028, 0x0030].value # 0028,0030 Pixel Spacing 0.742999970912933\0.742999970912933

    # display components
    ds.WindowCenter = [0]   # 0028,1050  Window Center
    ds.WindowWidth = [1]  # 0028,1051  Window Width
    ds.RescaleIntercept = 0  # 0028,1052  Rescale Intercept: 0
    ds.RescaleSlope = 1 # 0028,1053  Rescale Slope: 1

    ds.save_as(filename)


""" Image Stats / Display"""
def stat(array):
    #may need str casts?
    print('min: ' + np.min(array) + ' max: ' + np.max(array) + ' median: ' + np.median(array) + ' avg: ' + np.mean(array))


""" Image Preprocessing """
# nearest interpolation was the default and seems to preserve the mask image on resizes
def to_scale(img, shape=None):
    height, width = shape
    if img.dtype == SEG_DTYPE:
        # This function is only available if Python Imaging Library (PIL) is installed.
        # Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
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

def byte_normalize_image(img):
    """ Normalize image values to [0,255] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (255.0 * (img - min_) / (max_ - min_))

def normalize_image_using_rescale_slope_intercept(img, m, b):
    """ Normalize image values to y = mx + b """
    return ((m * img) + b)

def histeq_processor(img):
    """Histogram equalization"""
    nbr_bins=256
    #get image histogram
    #imhist,bins = np.histogram(img.flatten(),nbr_bins,normed=True)
    imhist, bins = np.histogram(img.flatten(),nbr_bins)  # keesh updated due to warning Deprecated since version 1.6.0.
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    #use linear interpolation of cdf to find new pixel values
    original_shape = img.shape
    img = np.interp(img.flatten(),bins[:-1],cdf)
    img=img/255.0
    return img.reshape(original_shape)


def step1_preprocess_img_slice(img_slc, slice, b, m, test_feature, results_dir):
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

    # must apply m and b first
    img_slc = normalize_image_using_rescale_slope_intercept(img_slc, m, b)

    # Should we increase the threshold range (if we are missing liver)?
    # CT body from MIS level=40, width=400  [-160, 240]
    # CT liver from IJ level 80, width=150  [5, 155]
    # CT abd from IJ level=50, width=350 [-125, 225]
    img_slc[img_slc>1200] = 0

    thresh_lo = -100
    thresh_hi = 400
    if first_time:
        print("thresh low  (HU)= " + str(thresh_lo))
        print("thresh high (HU)= " + str(thresh_hi))

    # Do we need to worry about VOI LUT Sequence (0028,3010) presennce in our CT images as this should get applied early.

    img_slc   = np.clip(img_slc, thresh_lo, thresh_hi)

    # If we apply auto WL convert back to np 16 bit (signed/unsigned) based on image data type read in (make sure that we are still in the 16-bit range after b/m)
    # then convert back to IMG_DTYPE.  Can we convert MIS Auto W/L to work with doubles?

    img_slc   = normalize_image(img_slc)  # [0,1]
    img_slc   = to_scale(img_slc, (388,388))
    img_slc   = np.pad(img_slc,((92,92),(92,92)),mode='reflect')

    return img_slc


def perform_inference(input_dir, results_dir, test_feature):
    """ Read Test Data """
    dcm_pattern = "*.dcm"
    if test_feature:
        dcm_pattern = "image_*"

    img, ds, num_images, b, m = read_dicom_series(input_dir + os.path.sep, filepattern=dcm_pattern)

    # process an image every every x slices
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    # Load network
    net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)
    print("step 1 net constructed")
    for slice_no in range(0, num_images):
        img_slice = img[..., slice_no]
        ds_slice = ds[slice_no]
        (num_rows, num_cols) = img_slice.shape

        # Prepare a test slice
        # May have to flip left to right (and change assumptions like HU thresholds)
        img_p = step1_preprocess_img_slice(img_slice, slice_no, b, m, test_feature, results_dir)

        """ Perform Inference """
        # Predict
        net1.blobs['data'].data[0,0,...] = img_p
        print("fed slice: " + str(slice_no))

        pred = net1.forward()['prob'][0,1] > 0.5
        print("pred1 mask: "+ str(slice_no))
        print("pred shape, type:" + str(pred.shape) + "," + str(type(pred)))

        #prepare step 1 output mask for saving
        mask1 = (pred > 0.5)  # [False, True]
        mask1 = mask1.astype(SEG_DTYPE)  # byte [0, 1]
        #resize using nearest to preserve mask shape
        mask1 = to_scale(mask1, (num_rows, num_cols))  # (512, 512)

        write_dicom_mask(mask1, ds_slice, slice_no, results_dir)

        if first_time:
            first_time = False
    # Free up memory of step1 network
    del net1  #needed ?


if __name__ == '__main__':
    '''
    This script runs step 1 of the Cascaded-FCN using its CT liver model on a test dicom dir.
    '''
    parser = argparse.ArgumentParser(description='step 1 of Cascaded-FCN test script')
    parser.add_argument("-i", dest="input_dicom_dir", help="The input dicom directory to read test images from")
    parser.add_argument("-o", dest="output_results_dir", help="The output directory to write results to")
    parser.add_argument("-t", "--test_feature", dest="test_feature", help="true or false. Whether to apply the current test feature")
    if len(sys.argv) < 5:
        print("python test_cascaded_unet_inference.py -i <input_dcm_dir> -o <output_results_dir> -t <true|false>")
        sys.exit(1)
    inpArgs = parser.parse_args()
    main(inpArgs)

