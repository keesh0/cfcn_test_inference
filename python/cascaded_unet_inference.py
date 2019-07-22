# Python 2.7.6 test script to test the following Caffe model on new CT liver images
# Translated from cascaded_unet_inference.ipynb to test CD'or 3.0 liver segmentation
# ref https://modelzoo.co/model/cascaded-fully-convolutional-networks-for-biomedical
# ref https://github.com/IBBM/Cascaded-FCN
import os
import sys
import wget  #keesh added to replace !wget

# Get model weights (step1 and step2 models)
# ref https://stackoverflow.com/questions/24346872/python-equivalent-of-a-given-wget-command
output_file1 = "../models/cascadedfcn/step1/step1_weights.caffemodel"
if os.path.isfile(output_file1) == False:
    url1 = "https://www.dropbox.com/s/aoykiiuu669igxa/step1_weights.caffemodel?dl=1"
    wget.download(url1, out=output_file1)
    print("Getting model1 weights: " + output_file1)
#

output_file2 = "../models/cascadedfcn/step2/step2_weights.caffemodel"
if os.path.isfile(output_file2) == False:
    url2 = "https://www.dropbox.com/s/ql10c37d7ura23l/step2_weights.caffemodel?dl=1"
    wget.download(url2, out=output_file2)
    print("Getting model2 weights: " + output_file2)

STEP1_DEPLOY_PROTOTXT = "../models/cascadedfcn/step1/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "../models/cascadedfcn/step1/step1_weights.caffemodel"
STEP2_DEPLOY_PROTOTXT = "../models/cascadedfcn/step2/step2_deploy.prototxt"
STEP2_MODEL_WEIGHTS   = "../models/cascadedfcn/step2/step2_weights.caffemodel"

import caffe
print("caffe__file: " + caffe.__file__)
# Use CPU for inference
# May get 1394 error -- hopefully not serious
# also Check failed: status == CUDNN_STATUS_SUCCESS (1 vs. 0)  CUDNN_STATUS_NOT_INITIALIZED
caffe.set_mode_cpu()

# Use GPU for inference -- does this ever work any any VM?
#caffe.set_mode_gpu()


import numpy as np
from matplotlib import pyplot as plt

from matplotlib import image as imsave
from IPython import display
plt.set_cmap('gray')
#888 from Jupyter Notebook code needed? -- %matplotlib inline

import scipy
import scipy.misc

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8

import dicom
import natsort
import glob
import re

import zipfile


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
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom

def read_liver_lesion_masks(masks_dirname):
    """Since 3DIRCAD provides an individual mask for each tissue type (in DICOM series format),
    we merge multiple tissue types into one Tumor mask, and merge this mask with the liver mask

    Args:
        masks_dirname : MASKS_DICOM directory containing multiple DICOM series directories,
                        one for each labelled mask
    Returns:
        numpy array with 0's for background pixels, 1's for liver pixels and 2's for tumor pixels
    """
    tumor_volume = None
    liver_volume = None

    # For each relevant organ in the current volume
    for organ in os.listdir(masks_dirname):
        organ_path = os.path.join(masks_dirname,organ)
        if not os.path.isdir(organ_path):
            continue

        organ = organ.lower()

        if organ.startswith("livertumor") or re.match("liver.yst.*", organ) or organ.startswith("stone") or organ.startswith("metastasecto") :
            print('Organ: ' + masks_dirname + " " + organ)
            current_tumor = read_dicom_series(organ_path)
            current_tumor = np.clip(current_tumor,0,1)
            # Merge different tumor masks into a single mask volume
            tumor_volume = current_tumor if tumor_volume is None else np.logical_or(tumor_volume,current_tumor)
        elif organ == 'liver':
            print('Organ: ' + masks_dirname + " " + organ)
            liver_volume = read_dicom_series(organ_path)
            liver_volume = np.clip(liver_volume, 0, 1)

    # Merge liver and tumor into 1 volume with background=0, liver=1, tumor=2
    label_volume = np.zeros(liver_volume.shape)
    label_volume[liver_volume==1]=1
    label_volume[tumor_volume==1]=2
    return label_volume


""" Image Stats / Display"""
def stat(array):
    #may need str casts?
    print('min: ' + np.min(array) + ' max: ' + np.max(array) + ' median: ' + np.median(array) + ' avg: ' + np.mean(array))


def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    if len(args) == 0:
        raise ValueError("No images given to imshow")
    elif len(args) == 1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n = len(args)
        if type(cmap) == str:
            cmap = [cmap]*n
        if type(title) == str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
    plt.show()


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


def preprocess_lbl_slice(lbl_slc):
    """ Preprocess ground truth slice to match output prediction of the network in terms
    of size and orientation.

    Args:
        lbl_slc: raw label/ground-truth slice
    Return:
        Preprocessed label slice"""
    lbl_slc = lbl_slc.astype(SEG_DTYPE)
    #downscale the label slc for comparison with the prediction
    lbl_slc = to_scale(lbl_slc , (388, 388))
    return lbl_slc

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


def step1_preprocess_img_slice(img_slc):
    """
    Preprocesses the image 3d volumes by performing the following :
    1- Rotate the input volume so the the liver is on the left, spine is at the bottom of the image
    2- Set pixels with hounsfield value great than 1200, to zero. 888
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
    img_slc[img_slc>1200] = 0
    img_slc   = np.clip(img_slc, -100, 400)
    # save HU image to disk
    imsave.imsave(results_dir + os.path.sep + 'preproc1hu_img_slice.png', img_slc)

    img_slc   = normalize_image(img_slc)
    img_slc   = to_scale(img_slc, (388,388))
    img_slc   = np.pad(img_slc,((92,92),(92,92)),mode='reflect')
    if False:
        img_slc = histeq_processor(img_slc)  #888 should we do this step?

    return img_slc


def step2_preprocess_img_slice(img_p, step1_pred):
    """ Preprocess img slice using the prediction image from step1, by performing
    the following :
    1- Set non-liver pixels to 0
    2- Calculate liver bounding box
    3- Crop the liver patch in the input img
    4- Resize (usually upscale) the liver patch to the full network input size 388x388
    5- Pad image slice with 92 on all sides

    Args:
        img_p: Preprocessed image slice
        step1_pred: prediction image from step1
    Return:
        The liver patch and the bounding box coordinate relative to the original img coordinates"""

    img = img_p[92:-92,92:-92]
    pred = step1_pred.astype(SEG_DTYPE)

    # Remove background !
    img = np.multiply(img,np.clip(pred,0,1))
    # get patch size
    col_maxes = np.max(pred, axis=0) # a row
    row_maxes = np.max(pred, axis=1) # a column

    nonzero_colmaxes = np.nonzero(col_maxes)[0]
    nonzero_rowmaxes = np.nonzero(row_maxes)[0]

    x1, x2 = nonzero_colmaxes[0], nonzero_colmaxes[-1]
    y1, y2 = nonzero_rowmaxes[0], nonzero_rowmaxes[-1]
    width = x2-x1
    height= y2-y1
    MIN_WIDTH = 60
    MIN_HEIGHT= 60
    x_pad = (MIN_WIDTH - width) / 2 if width < MIN_WIDTH else 0
    y_pad = (MIN_HEIGHT - height)/2 if height < MIN_HEIGHT else 0

    x1 = max(0, x1-x_pad)
    x2 = min(img.shape[1], x2+x_pad)
    y1 = max(0, y1-y_pad)
    y2 = min(img.shape[0], y2+y_pad)

    img = img[y1:y2+1, x1:x2+1]
    pred = pred[y1:y2+1, x1:x2+1]

    img = to_scale(img, (388,388))
    pred = to_scale(pred, (388,388))
    # All non-lesion is background
    pred[pred==1]=0
    # Lesion label becomes 1
    pred[pred==2]=1

    # Now do padding for UNET, which takes 572x572
    #pred=np.pad(pred,((92,92),(92,92)),mode='reflect')
    img=np.pad(img,92,mode='reflect')
    return img, (x1,x2,y1,y2)


""" Get Test Data (should extract and pass to a new wrapper function) """
# Download image 17 of 3DIRCAdb1 dataset
url3 = "http://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.17.zip"
output_file3 = "3Dircadb1.17.zip"
if os.path.isfile(output_file3) == False:
    wget.download(url3, out=output_file3)
    print("Got: " + os.getcwd() + os.path.sep + output_file3)

# Unzip into test_image
if not os.path.isdir('test_image') or len(os.listdir('test_image')) == 0:
    with zipfile.ZipFile("3Dircadb1.17.zip","r") as zip_ref:
        zip_ref.extractall("test_image")
        print("Extracted: test_image/3Dircadb1.17.zip")

# Unzip the image CT volume
with zipfile.ZipFile("test_image/3Dircadb1.17/PATIENT_DICOM.zip","r") as zip_ref:
    zip_ref.extractall("test_image/3Dircadb1.17/")
    print("Extracted test_image/3Dircadb1.17/PATIENT_DICOM.zip")
# Unzip the label masks
with zipfile.ZipFile("test_image/3Dircadb1.17/MASKS_DICOM.zip","r") as zip_ref:
    zip_ref.extractall("test_image/3Dircadb1.17/")
    print("Extracted test_image/3Dircadb1.17/MASKS_DICOM.zip")

""" Read Test Data """
img = read_dicom_series("test_image/3Dircadb1.17/PATIENT_DICOM/")

# optional step
lbl = np.zeros(0)
lbl = read_liver_lesion_masks("test_image/3Dircadb1.17/MASKS_DICOM/")

""" Show Test Data """
print("Test image shape: " + str(img.shape) + " Label shape: " + str(lbl.shape))
# for s in range(50,100,20):
#     imshow(img[...,s],lbl[...,s])
# for s in range(50,100,20):
#     print("Test slice #: " + str(s))
#     img_p = step1_preprocess_img_slice(img[...,s])
#     lbl_p = preprocess_lbl_slice(lbl[...,s])
#     imshow(img_p,lbl_p)


""" Perform Inference """
# Prepare a test slice
results_dir = 'test_image' + os.path.sep + "results"
os.mkdir(results_dir)
S = 90
img_p = step1_preprocess_img_slice(img[...,S])

lbl_p = np.zeros(0)
if lbl.size != 0:
    lbl_p = preprocess_lbl_slice(lbl[...,S])
# Show preprocessed test slice
#imshow(img_p,lbl_p,title=['Test image','Ground truth'])
#May have to scale the intensities and flip left to rght
imsave.imsave(results_dir + os.path.sep + 'preproc1_img_slice.png', img_p)
if lbl_p.size != 0:
    imsave.imsave(results_dir+ os.path.sep + 'preproc_lbl_slice.png', lbl_p)

# Finally, the machine learning piece!
# Load network
net1 = caffe.Net(STEP1_DEPLOY_PROTOTXT, STEP1_MODEL_WEIGHTS, caffe.TEST)
print("net1 constructed")

# Predict
net1.blobs['data'].data[0,0,...] = img_p
print("image data fed")
pred = net1.forward()['prob'][0,1] > 0.5
print("prediction generated")
print(pred.shape)

# Visualize results
#imshow(img_p, lbl_p, pred>0.5, title=['Slice','Ground truth', 'Prediction'])
imsave.imsave(results_dir+ os.path.sep + 'pred1_slice.png', (pred>0.5))

# Free up memory of step1 network
del net1

# Prepare liver patch for step2
# net1 output is used to determine the predicted liver bounding box
img_p2, bbox = step2_preprocess_img_slice(img_p, pred)
#imshow(img_p2)
imsave.imsave(results_dir+ os.path.sep + 'preproc2_img_slice.png', img_p2)

# Load step2 network
net2 = caffe.Net(STEP2_DEPLOY_PROTOTXT, STEP2_MODEL_WEIGHTS, caffe.TEST)

# Predict
net2.blobs['data'].data[0,0,...] = img_p2
pred2 = net2.forward()['prob'][0,1]
print("Predicted stage2 shape: " + str(pred2.shape))

# Visualize result
# extract liver portion as predicted by net1
imsave.imsave(results_dir+ os.path.sep + 'pred2_slice.png', (pred2>0.5))
imsave.imsave(results_dir+ os.path.sep + 'preproc22_img_slice.png', img_p2[92:-92,92:-92])
lbl_p_liver = np.zeros(0)
if lbl_p.size != 0:
    x1,x2,y1,y2 = bbox
    lbl_p_liver = lbl_p[y1:y2,x1:x2]
    # Set labels to 0 and 1
    lbl_p_liver[lbl_p_liver==1]=0
    lbl_p_liver[lbl_p_liver==2]=1
    #imshow(img_p2[92:-92,92:-92],lbl_p_liver, pred2>0.5)
    imsave.imsave(results_dir+ os.path.sep + 'lbl22_slice.png', lbl_p_liver)



