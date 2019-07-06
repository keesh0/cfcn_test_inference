# Python 2.7.x test script to test the 3D CRF model on our CFCN step 1 liver masks.
# ref https://modelzoo.co/model/cascaded-fully-convolutional-networks-for-biomedical
# ref https://github.com/mbickel/DenseInferenceWrapper
import os
import sys
import argparse

import numpy as np
from numpy import zeros, newaxis

# cheap method to find 3DCRF Python 2.7 package under AWS Python 2.7.x caffe env Ubuntu
sys.path.append('/usr/local/lib/python2.7/dist-packages')
print('\n'.join(sys.path))
from denseinference import CRFProcessor

from test_cascaded_unet_inference import read_dicom_series
from test_cascaded_unet_inference import write_dicom_mask
from test_cascaded_unet_inference import stat
from test_cascaded_unet_inference import normalize_image
from test_cascaded_unet_inference import normalize_image_using_rescale_slope_intercept
from test_cascaded_unet_inference import IMG_DTYPE

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


def step3_preprocess_img_slice(img_slc, b, m):
    """
    Preprocesses the image 3d volumes by performing the following :
    1- Normalize values to [0, 1]

    Args:
        img_slc: raw image slice
    Return:
        Preprocessed image slice
    """
    img_slc = img_slc.astype(IMG_DTYPE)

    # must apply m and b first
    img_slc = normalize_image_using_rescale_slope_intercept(img_slc, m, b)
    img_slc = normalize_image(img_slc)  # [0,1]
    return img_slc


def perform_3dcrf(input_dir, mask_dir, results_dir, test_feature):
    """ Read Test Data """
    dcm_pattern = "*.dcm"
    if test_feature:
        dcm_pattern = "image_*"

    print('start perform 3dcrf...')
    img, ds, num_images, b, m = read_dicom_series(input_dir + os.path.sep, filepattern=dcm_pattern)  # rows, cols, slices
    (num_rows, num_cols, num_slices) = img.shape
    print("Read from: " + input_dir)

    ConstPixelDims = (num_rows, num_cols, num_slices)
    img_array = np.zeros(ConstPixelDims, dtype=img.dtype)
    for slice_no in range(0, num_slices):
        img_slice = img[..., slice_no]
        # normalize input image to [0.1]
        img_p = step3_preprocess_img_slice(img_slice, b, m)
        img_array[:, :, slice_no] = img_p

    print('Preprocessed input images')

    # 3DCRF Image: (W, H, D)
    crf_img  = np.swapaxes(img_array, 0, 1)
    print("crf image shape: " + str(crf_img.shape))

    mask_pattern = "*_mask1.dcm"
    mask, ds_mask, num_mask_images, b, m = read_dicom_series(mask_dir + os.path.sep, filepattern=mask_pattern)  # rows, cols, slices
    # Label (W, H, D, C)
    crf_mask  = np.swapaxes(mask, 0, 1)
    # Convert 3D mask to 4D tensor
    feature_tensor = crf_mask[..., newaxis]
    print("crf feature tensor shape: " + str(feature_tensor.shape))

    # 3D CRF
    pro = CRFProcessor.CRF3DProcessor()
    print("step 3 CRF3DProcessor constructed")

    # Now run crf and get hard labeled result tensor:
    # Hard labeled result as ndarray. (W, H, D), [0, L], dtype=int16
    result = pro.set_data_and_run(crf_img, feature_tensor)
    print("step 3 CRF3DProcessor complete.")

    (W, H, D) = result.shape
    print("crf result shape: " + str(result.shape))

    mask2  = np.swapaxes(result, 0, 1)
    print("mask2 shape: " + str(mask2.shape))

    # Write out resultant mask2 as DICOM
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    for slice_no in range(0, D):
        mask2_slice = mask2[..., slice_no]
        ds_slice = ds[slice_no]
        write_dicom_mask(mask2_slice, ds_slice, slice_no, results_dir, mask_suffix="_mask2")


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
