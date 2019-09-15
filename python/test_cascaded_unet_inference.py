# Test script to segment CT liver images.
# Python 3.6.8, Caffe 1.0.0, CUDA 10.1 (V10.1.243), numpy 1.17.2, scipy 1.2.0 (last one for imresize)
# dicom 0.9.9-1 (should upgrade to pydicom), nibabel 2.5.0
# Translated from cascaded_unet_inference.ipynb to test CD'or 3.0 liver segmentation
# I live under https://github.com/keesh0/cfcn_test_inference
# ref https://modelzoo.co/model/cascaded-fully-convolutional-networks-for-biomedical
# ref https://github.com/IBBM/Cascaded-FCN
import os
import sys
from pathlib import Path
import wget

import argparse

import caffe

import numpy as np
import scipy
import scipy.misc

import dicom
from dicom.dataset import FileDataset
import nibabel as nib
import nibabel.orientations as orientations

import natsort
import glob

#globals for now
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
STEP1_DEPLOY_PROTOTXT = PROJECT_DIRECTORY + os.path.sep + "models" + os.path.sep + "cascadedfcn" + os.path.sep + "step1" + os.path.sep + "step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = PROJECT_DIRECTORY + os.path.sep + "models" + os.path.sep + "cascadedfcn" + os.path.sep + "step1" + os.path.sep + "step1_weights.caffemodel"
IMG_DTYPE = np.float  # same as float64
SEG_DTYPE = np.uint8
MASK_DTYPE = np.uint16

def main(inpArgs):
    try:
        # Get model weights (step1 models)
        # ref https://stackoverflow.com/questions/24346872/python-equivalent-of-a-given-wget-command
        output_file1 = STEP1_MODEL_WEIGHTS
        if os.path.isfile(output_file1) == False:
            url1 = "https://www.dropbox.com/s/aoykiiuu669igxa/step1_weights.caffemodel?dl=1"
            print("Getting model 1 weights: " + output_file1)
            wget.download(url1, out=output_file1)

        print("caffe__file: " + caffe.__file__)
        # Use CPU for inference
        # caffe.set_mode_cpu()
        # Use GPU for inference -- Need exact CUDA version/driver synched with Caffe!
        caffe.set_mode_gpu()
        caffe.set_device(0)  #needed?

        perform_inference(os.path.abspath(inpArgs.input_dir_file), os.path.abspath(inpArgs.output_results_dir))

        sys.exit(0)
    except IOError as ioex:
        print("There was an IO error: " + str(ioex.strerror))
        sys.exit(1)

    except Exception as e:
        print("There was an unexpected error: " + str(e))
        sys.exit(1)


""" Image I/O  """
def read_dicom_series(directory, filepattern = "*.dcm"):
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
        # DICOM type 1 required
        b = float(ds[0x0028, 0x1052].value)  # 0028,1052  Rescale Intercept: -1024
        m = float(ds[0x0028, 0x1053].value)  # 0028,1053  Rescale Slope: 1

    return ArrayDicom, Arrayds, len(lstFilesDCM), b, m

def write_dicom_mask(img_slice, ds, slice_no, outputdirectory, mask_suffix, filepattern = ".dcm"):
    ds_slice = ds[slice_no]
    series_number = ds_slice[0x0020, 0x0011].value
    base_fname = str(slice_no).zfill(6)
    filename = outputdirectory + os.path.sep + base_fname + "_" + str(series_number) + mask_suffix + filepattern
    ds = FileDataset(filename, ds_slice, file_meta=ds_slice.file_meta, preamble=b"\0" * 128)

    ds.SOPInstanceUID = ds_slice.SOPInstanceUID
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    ds.StudyID = "123"
    ds.PatientName = "Liver^Larry^H"

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # These are the necessary imaging components of the FileDataset object.
    (rows, cols) = img_slice.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.Columns = cols
    ds.Rows = rows
    ds.PixelData = img_slice.tobytes()

    image_type_val = ds_slice[0x0008, 0x0008].value
    image_type_val_str = "\\".join(str(x) for x in image_type_val)
    image_type_val_str2 = image_type_val_str.replace("ORIGINAL", "DERIVED", 1)
    ds.ImageType = image_type_val_str2

    # display components
    ds.WindowCenter = [0]   # 0028,1050  Window Center
    ds.WindowWidth = [1]  # 0028,1051  Window Width
    ds.RescaleIntercept = 0  # 0028,1052  Rescale Intercept: 0
    ds.RescaleSlope = 1 # 0028,1053  Rescale Slope: 1

    ds.save_as(filename)

def read_nifti_series(filename):
    proxy_img = nib.load(filename)
    # less efficent get image data into memory all at once
    # image_data = proxy_img.get_fdata()

    hdr = proxy_img.header
    (num_rows, num_cols, num_images, _) = hdr.get_data_shape()  # not sure of this order
    (m, b) = hdr.get_slope_inter()
    axcodes = nib.aff2axcodes(proxy_img.affine)
    if (axcodes != ('R', 'A', 'S')) and (axcodes != ('L', 'A', 'S')):
        print("Input NIfti series is in unsupported orientation.  Please convert to RAS or LAS orientation:" + filename)
        sys.exit(1)

    # specifiy LPS for DICOM
    # https://nipy.org/nibabel/dicom/dicom_orientation.html, if we want to apply the formula above to array indices in pixel_array, we first have to apply a column / row flip to the indices.
    codes = ('L', 'P', 'S')
    labels = (('A','P'),('R','L'),('I','S'))
    orients = orientations.axcodes2ornt(codes, labels)
    img_reorient = proxy_img.as_reoriented(orients)
    hdr = img_reorient.header
    # We reset m and b here ourselves for downstream rescale/slope
    b = 0
    m = 1
    return img_reorient, hdr, num_images, b, m, axcodes

# img_reorient is the orig input NIFTI image in DICOM LPS
def write_nifti_mask(img_reorient, axcodes, mask_data, outputdirectory, base_fname, filepattern = ".nii"):
    # We only currently support NIfti LAS and RAS orientations
    # TODO--  To support more NIfti orientations add more key/values to nifti_in_codes_labels
    nifti_in_codes_labels =	{
        ('L', 'A', 'S'): (('P','A'),('R','L'),('I','S')),
        ('R', 'A', 'S'): (('P','A'),('L','R'),('I','S'))
    }
    filename = outputdirectory + os.path.sep + base_fname + "_mask1" + filepattern
    new_header = header = img_reorient.header.copy()
    new_header.set_slope_inter(1, 0)  # no scaling
    new_header['cal_min'] = np.min(mask_data)
    new_header['cal_max'] = np.max(mask_data)
    new_header['bitpix'] = 16
    new_header['descrip'] = "NIfti mask volume from Caffe 1.0"

    mask2 = np.zeros(img_reorient.shape, MASK_DTYPE)
    mask2[..., 0] = mask_data  # Add a 4th dim for Nifti, not sure if last dim is number of channels?
    nifti_mask_img = nib.nifti1.Nifti1Image(mask2, img_reorient.affine, header=new_header)

    # Need to xform numpy from supposed DICOM LPS to NIFTI original orientation (i.e. LAS, RAS, etc.)
    orients = orientations.axcodes2ornt(axcodes, nifti_in_codes_labels[axcodes])
    mask_reorient = nifti_mask_img.as_reoriented(orients)
    nib.save(mask_reorient, filename)

# returns false for a probable NIfti file and true for a possible DICOM dir/file
def test_load_as_dicom(path):
    if os.path.isfile(path) and Path(path).suffix == ".nii":
        return False
    return True

def get_image_slice(img, slice_no, load_as_dicom):
    if load_as_dicom:
        return img[..., slice_no]
    else:
        return img.dataobj[..., slice_no, 0]

""" Image Stats / Display"""
def stat(array):
    print('min: ' + str(np.min(array)) + ' max: ' + str(np.max(array)) + ' median: ' + str(np.median(array)) + ' avg: ' + str(np.mean(array)))


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
    elif img.dtype == MASK_DTYPE:
        max_ = np.max(img)
        factor = 255.0/max_ if max_ != 0 else 1
        return (scipy.misc.imresize(img,(height,width),interp="nearest")/factor).astype(MASK_DTYPE)
    else:
        raise TypeError('Error. To scale the image array, its type must be np.uint8 or np.float64 or np.uint16. (' + str(img.dtype) + ')')

def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)

def normalize_image_using_rescale_slope_intercept(img, m, b):
    """ Normalize image values to y = mx + b """
    return ((m * img) + b)

def step1_preprocess_img_slice(img_slc, slice, b, m, results_dir):
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

    # must apply m and b first for DICOM, for NIFTI its always 1,0
    img_slc = normalize_image_using_rescale_slope_intercept(img_slc, m, b)

    # Should we increase the threshold range (if we are missing liver)?
    # CT body from MIS level=40, width=400  [-160, 240]
    # CT liver from IJ level 80, width=150  [5, 155]
    # CT abd from IJ level=50, width=350 [-125, 225]
    img_slc[img_slc>1200] = 0

    thresh_lo = -100
    thresh_hi = 400

    # Do we need to worry about VOI LUT Sequence (0028,3010) presennce in our CT images as this should get applied early.

    img_slc   = np.clip(img_slc, thresh_lo, thresh_hi)

    img_slc   = normalize_image(img_slc)  # [0,1]
    img_slc   = to_scale(img_slc, (388,388))
    img_slc   = np.pad(img_slc,((92,92),(92,92)),mode='reflect')

    return img_slc


def perform_inference(input_dir_file, results_dir):
    """ Read Test Data """
    load_as_dicom = test_load_as_dicom(input_dir_file)
    if load_as_dicom:
        img, ds, num_images, b, m = read_dicom_series(input_dir_file + os.path.sep)
    else:
        img, hdr, num_images, b, m, axcodes = read_nifti_series(input_dir_file)

    # process an image every every x slices
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    # Load network Caffe 1.0.0
    net1 = caffe.Net(network_file=STEP1_DEPLOY_PROTOTXT, weights=STEP1_MODEL_WEIGHTS, phase=caffe.TEST)
    # Load network pre Caffe 1.0.0
    # net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)
    print("step 1 net constructed")
    mask_data_array = None  # NIFTI only
    for slice_no in range(0, num_images):
        img_slice = get_image_slice(img, slice_no, load_as_dicom)
        (num_rows, num_cols) = img_slice.shape

        # Prepare a test slice
        # May have to flip left to right (and change assumptions like HU thresholds)
        img_p = step1_preprocess_img_slice(img_slice, slice_no, b, m, results_dir)

        """ Perform Inference """
        # Predict
        net1.blobs['data'].data[0,0,...] = img_p

        # take the first dim of 'prob' index 0, second dim of 'prob' index 1, and all of  the third dim
        pred = net1.forward()['prob'][0,1] > 0.5

        #prepare step 1 output mask for saving
        mask1 = (pred > 0.5)  # [False, True]
        mask1 = mask1.astype(MASK_DTYPE)  # uint16 [0, 1]
        #resize using nearest to preserve mask shape
        mask1 = to_scale(mask1, (num_rows, num_cols))  # (512, 512)

        if load_as_dicom:
            write_dicom_mask(mask1, ds, slice_no, results_dir, mask_suffix="_mask1")
        else:
            # NIFTI create and fill mask array
            if slice_no == 0:
                ConstMaskDims = (num_rows, num_cols, num_images)
                mask_data_array = np.zeros(ConstMaskDims, dtype=MASK_DTYPE)
            mask_data_array[..., slice_no] = mask1
        print("Processed slice: " + str(slice_no+1) + " of " + str(num_images))

    # Free up memory of step1 network
    del net1  #needed ?

    if not load_as_dicom:
        nifti_base = Path(input_dir_file).resolve().stem
        write_nifti_mask(img, axcodes, mask_data_array, results_dir, nifti_base)

    print("Caffe liver inference COMPLETE.")

if __name__ == '__main__':
    '''
    This script runs step 1 of the Cascaded-FCN using its CT liver model on a test dicom dir.
    '''
    parser = argparse.ArgumentParser(description='step 1 of Cascaded-FCN test script')
    parser.add_argument("-i", dest="input_dir_file", help="The input directory of dicom files to read test images from or the complete path to a NIfti1 format test file")
    parser.add_argument("-o", dest="output_results_dir", help="The output directory to write results to")
    if len(sys.argv) < 4:
        print("python test_cascaded_unet_inference.py -i <input_dcm_dir> -o <output_results_dir>")
        parser.print_help()
        sys.exit(1)
    inpArgs = parser.parse_args()
    main(inpArgs)
