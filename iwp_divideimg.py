#!/usr/bin/env python3

import os, os.path
from osgeo import gdal
import numpy as np
from gdalconst import *
import cv2

def divide_image(input_image_path,    # the image directory
                 output_divided_root, # the output directory for divided images
                 target_blocksize):   # the crop size

    # -------------------- read image information ---------------

    # read image
    input_image_dataset = gdal.Open(input_image_path)

    # extract image information
    proj = input_image_dataset.GetProjection()
    rows_input = input_image_dataset.RasterYSize
    cols_input = input_image_dataset.RasterXSize
    bands_input = input_image_dataset.RasterCount

    # create empty grid cell
    transform = input_image_dataset.GetGeoTransform()

    # ulx, uly is the upper left corner
    ulx, x_resolution, _, uly, _, y_resolution  = transform

    # get 4, 3, 2 bands
    band_4_raster = input_image_dataset.GetRasterBand(4)
    band_3_raster = input_image_dataset.GetRasterBand(3)
    band_2_raster = input_image_dataset.GetRasterBand(2)

    # ---------------------- Divide image ----------------------
    overlap_rate = 0.2
    block_size = target_blocksize
    ysize = rows_input
    xsize = cols_input

    # ---------------------- Find each Upper left (x,y) for each images ----------------------
    for i in range(0, ysize, int(block_size*(1-overlap_rate))):

        # don't want moving window to be larger than row size of input raster
        if i + block_size < ysize:  
            rows = block_size  
        else:  
            rows = ysize - i

        # read col      
        for j in range(0, xsize, int(block_size*(1-overlap_rate))):

            if j + block_size < xsize:  
                    cols = block_size  
            else:  
                cols = xsize - j 

            # get block out of the whole raster
            band_4_array = band_4_raster.ReadAsArray(j, i, cols, rows) 
            band_3_array = band_3_raster.ReadAsArray(j, i, cols, rows)
            band_2_array = band_2_raster.ReadAsArray(j, i, cols, rows)

            # filter out black image
            if band_4_array[0,0] == 0 and band_4_array[0,-1] == 0 and  \
               band_4_array[-1,0] == 0 and band_4_array[-1,-1] == 0:
                continue

            # stack three bands into one array
            output_array = np.stack((band_2_array, band_3_array, band_4_array), axis=2)

            # Upper left (x,y) for each images
            ul_row_divided_img = uly + i*y_resolution
            ul_col_divided_img = ulx + j*x_resolution

            # setup the output path
            output_jpg_name = "%s_%s_%s_%s.jpg"%(i,j,ul_row_divided_img,ul_col_divided_img)
            output_jpg_path = os.path.join(output_divided_root,output_jpg_name)

            if os.path.exists(output_jpg_path):
                print ("Divided images exsited: ", output_jpg_name)
                continue
            else:
                cv2.imwrite(output_jpg_path,output_array)
                print ("Dividing image: ", output_jpg_name)

    return x_resolution, y_resolution
