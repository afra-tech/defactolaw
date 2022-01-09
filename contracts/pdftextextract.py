import pdf2image
from pdf2image import convert_from_path
import cv2
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None
import cv2
from numpy import byte, bytes_
import pytesseract
from PIL import Image
import glob
import os
import img2pdf
import fitz
import PIL.Image
from os import listdir

import os
import sys
import shutil
import gc


thisfolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tesspath = os.path.join(thisfolder, 'tessdata')

#tessdata_dir_config = '--tessdata-dir "'+tesspath+'"'

def getscannedtext(pdf):



    return ocrextract(pdf)




def convert_pdf_to_img(pdf_file):
    """
    @desc: this function converts a PDF into Image

    @params:
        - pdf_file: the file to be converted

    @returns:
        - an interable containing image format of all the pages of the PDF
    """
    return convert_from_path(pdf_file, poppler_path=r'')

def ocrextract(pdf):
    # Directory
    directory = "tempFolder"

    # Parent Directory path
    THIS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abspdfpath = os.path.join(THIS_FOLDER, 'output.pdf')


    # image_path = "tempFolder/images/"tempFolder/images/
    inpath = pdf
    images = convert_pdf_to_img(inpath)
    result = fitz.open()
    temppdfpath = os.path.join(THIS_FOLDER, 'temppdf.pdf')

    for n, image in enumerate(images):
        pdf = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
        with open(temppdfpath, 'w+b') as f:
            f.write(pdf)
        del pdf
        del image
        gc.collect()
        with fitz.open(temppdfpath) as mfile:
            result.insertPDF(mfile)

        #image.save(image_path + 'imageSample' + str(n) + '.jpg')

    result.save(abspdfpath)
    del result
    gc.collect()
    try:
        os.remove(temppdfpath)
    except OSError:
        pass
 