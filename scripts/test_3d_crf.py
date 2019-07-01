# Python 2.7.6 test script to test the following 3D CRF model on our CT liver images
# ref https://modelzoo.co/model/cascaded-fully-convolutional-networks-for-biomedical
# ref https://github.com/mbickel/DenseInferenceWrapper
import os
import argparse

import numpy as np
from numpy import zeros, newaxis

import dicom
from dicom.dataset import Dataset
from dicom.dataset import FileDataset
import datetime
import time
import platform
import natsort
import glob

import sys
# cheap method to find 3DCRF Python 2.7 package under Ubuntu
sys.path.append('/usr/local/lib/python2.7/dist-packages')
print('\n'.join(sys.path))
from denseinference import CRFProcessor

#globals for now
IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

def main(inpArgs):
    try:
        test_feature = False
        if inpArgs.test_feature.lower() == "true":  # "true" or "false" as a string
            test_feature = True

        perform_3dcrf(os.path.abspath(inpArgs.input_dicom_dir), os.path.abspath(inpArgs.input_mask_dir),
                      os.path.abspath(inpArgs.output_results_dir), test_feature)

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
        # DICOM type 1 required
        b = float(ds[0x0028, 0x1052].value)  # 0028,1052  Rescale Intercept: -1024
        m = float(ds[0x0028, 0x1053].value)  # 0028,1053  Rescale Slope: 1

    return ArrayDicom, Arrayds, len(lstFilesDCM), b, m

def write_dicom_mask2(img_slice, ds_slice, slice_no, outputdirectory, filepattern = ".dcm"):
    file_meta = Dataset()
    #will need to generate all UID  uniqly see Mayo Image Studio
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'

    series_number = ds_slice[0x0020, 0x0011].value
    base_fname = str(slice_no).zfill(6)
    filename = outputdirectory + os.path.sep + base_fname + "_" + str(series_number) + "_mask2" + filepattern
    ds = FileDataset(filename, {}, file_meta = file_meta, preamble=b"\0"*128)
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
    ds.SmallestImagePixelValue = b'\\x00\\x00'
    ds.LargestImagePixelValue = b'\\x01\\x01'
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
def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)

def normalize_image_using_rescale_slope_intercept(img, m, b):
    """ Normalize image values to y = mx + b """
    return ((m * img) + b)

def step3_preprocess_img_slice(img_slc, b, m):
    """
    Preprocesses the image 3d volumes by performing the following :
    1- Normalize values to [0, 1]

    Args:
        img_slc: raw image slice
    Return:
        Preprocessed image slice
    """
    img_slc   = img_slc.astype(IMG_DTYPE)

    # must apply m and b first
    img_slc = normalize_image_using_rescale_slope_intercept(img_slc, m, b)
    img_slc = normalize_image(img_slc)  # [0,1]
    return img_slc


def perform_3dcrf(input_dir, mask_dir, results_dir, test_feature):
    """ Read Test Data """
    dcm_pattern = "*.dcm"
    if test_feature:
        dcm_pattern = "image_*"

    # Image: (W, H, D);
    # 888 switch row/col order
    img, ds, num_images, b, m = read_dicom_series(input_dir + os.path.sep, filepattern=dcm_pattern)  # rows, cols, slices

    (num_rows, num_cols, num_slices) = img.shape

    ConstPixelDims = (num_rows, num_cols, num_slices)
    # The array is sized based on 'ConstPixelDims'
    img_array = np.zeros(ConstPixelDims, dtype=img.dtype)
    for slice_no in range(0, num_images):
        img_slice = img[..., slice_no]
        img_p = step3_preprocess_img_slice(img_slice, b, m)
        img_array[:, :, slice_no] = img_p
    # Need to tranpose rows & cols ???

    # Label (W, H, D, C)
    mask_pattern = "*_mask1.dcm"
    mask, ds_mask, num_mask_images = read_dicom_series(mask_dir + os.path.sep, filepattern=mask_pattern)  # rows, cols, slices
    # 888 switch row/col order

    # Convert 3D mask to 4D tensor
    feature_tensor = mask[..., newaxis]
    print(feature_tensor.shape)

    # process an image every every x slices
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # 3D CRF
    pro = CRFProcessor.CRF3DProcessor()
    print("step 3 CRF3DProcessor constructed")

    # Now run crf and get hard labeled result tensor:
    result = pro.set_data_and_run(img, feature_tensor)
    # Result tensor is hard labeled (W, H, D)
    (W, H, D) = result.shape

    # Write out resultant mask2 as DICOM
    # 888 switch row/col order
    for slice_no in range(0, D):
        mask2_slice = result[..., slice_no]
        ds_slice = ds[slice_no]
        write_dicom_mask2(mask2_slice, ds_slice, slice_no, results_dir)

if __name__ == '__main__':
    '''
    This script runs step 3 (DenseCRF) of the Cascaded-FCN using on a test dicom dir.
    '''
    parser = argparse.ArgumentParser(description='step 3 of Cascaded-FCN test script')
    parser.add_argument("-i", dest="input_dicom_dir", help="The input dicom directory to read test images from")
    parser.add_argument("-m", dest="input_mask_dir", help="The input mask directory to read test masks from (soft-labeled classifier outputs)")
    parser.add_argument("-o", dest="output_results_dir", help="The output directory to write results to")
    parser.add_argument("-t", "--test_feature", dest="test_feature", help="true or false. Whether to apply the current test feature")
    if len(sys.argv) < 5:
        print("python test_3d_crf.py -i <input_dcm_dir> -m <input_mask_dir> -o <output_results_dir> -t <true|false>")
        sys.exit(1)
    inpArgs = parser.parse_args()
    main(inpArgs)

