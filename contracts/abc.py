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


thisfolder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tesspath = os.path.join(thisfolder, 'tessdata')

tessdata_dir_config = '--tessdata-dir "'+tesspath+'"'

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
    parent_dir =  THIS_FOLDER
    # Path
    path = os.path.join(parent_dir, directory)

    # Create the directory
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    os.mkdir(path)

    image_path = os.path.join(path, "images")
    os.mkdir(image_path)
    pdf_path = os.path.join(path, "pdf")
    os.mkdir(pdf_path)

    # image_path = "tempFolder/images/"tempFolder/images/
    inpath = pdf
    images = convert_pdf_to_img(inpath)

    for n, image in enumerate(images):
        image.save(image_path + 'imageSample' + str(n) + '.jpg')

    result = bytearray()
    input_dir = image_path

    for image_path in os.listdir(input_dir):
        input_path = os.path.join(input_dir, image_path)

        img = cv2.imread(input_path, 1)
        result += bytearray(pytesseract.image_to_pdf_or_hocr(img, lang='eng', config=tessdata_dir_config))

        name = os.path.basename(image_path).split('.')[0]
        with open(pdf_path + name + ".pdf", 'w+b') as f:
            f.write(bytearray(result))
        f.close()

    ##to merge pipe isntall mupdf

    output_dir = pdf_path

    result = fitz.open()

    for pdf in os.listdir(output_dir):
        output_path = os.path.join(output_dir, pdf)
        with fitz.open(output_path) as mfile:
            result.insertPDF(mfile)


    try:
        os.remove(abspdfpath)
    except OSError:
        pass
    result.save(abspdfpath)
    # delete the folder

    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))