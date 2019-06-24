#!/usr/bin/env python3

"""
Author: Weixing Zhang
Starting date: 2018-08-31 
Processing IWP mapping on HPC

Perform a large number of independent mapping tasks 
when there are more tasks than processors, especially
when the run times vary for each task.

Citation: This example was contributed by Craig Finch <cfinch@ieee.org and 
                                          Fernandoo Paolo <fpaolo@ucsd.edu>.
          Inspired by http://math.acadiau.ca/ACMMaC/Rmpi/index.html and 
                      https://gist.github.com/fspaolo/51eaf5a20d6d418bd4d0
"""

from __future__ import print_function
from mpi4py import MPI
from time import sleep 
from random import randint
from queue import Queue

import os, os.path
import shutil

import iwp_divideimg as divide
import iwp_inferenceimg as inference
import iwp_stitchshpfile as stitch


# work tag
WORKTAG = 1
DIETAG = 0


class Work(object):
    #put all work in queue to consume
    
    def __init__(self, files):
        # importat: sort by file size in decreasing order!
        files.sort(key=lambda f: os.stat(f).st_size, reverse=True)
        q = Queue()
        for f in files:
            q.put(f)
        self.work = q

    def get_next(self):
        if self.work.empty():
            return None
        return self.work.get()


def processing_img(rank,name,work):
    
    # _______________________SETTING_____________________________
    crop_size = 600

    # master root
    master_img_root = r"/pylon5/ps5fp1p/wez13005/local_dir/datasets/polygon/input_img/"
    
    # worker root
    worker_root = r"/pylon5/ps5fp1p/wez13005/local_dir/datasets/polygon/"
    worker_img_root = os.path.join(worker_root,"input_img")
    worker_divided_img_root = os.path.join(worker_root,"divided_img")
    worker_output_shp_root = os.path.join(worker_root,"output_shp")
    worker_finaloutput_root = os.path.join(worker_root,"final_shp")
    
    # path to the whole image in the worker node
    print ("Start processing image: ", work)

    input_img_name = work.split('/')[-1]
    input_img_path = os.path.join(worker_img_root,input_img_name)
    
    # path in the module
    POLYGON_DIR = worker_root
    weights_path = r"/pylon5/ps5fp1p/wez13005/local_dir/logs/ice_wedge_polygon20180823T1403/mask_rcnn_ice_wedge_polygon_0008.h5"

    """ Create subfolder for each image
    """
    worker_divided_img_subroot = os.path.join(worker_divided_img_root, input_img_name.split('.tif')[0])
    worker_output_shp_subroot = os.path.join(worker_output_shp_root, input_img_name.split('.tif')[0])
    worker_finaloutput_subroot = os.path.join(worker_finaloutput_root, input_img_name.split('.tif')[0])

    try:
        # shutil.rmtree(worker_img_root)
        shutil.rmtree(worker_divided_img_subroot)
        shutil.rmtree(worker_output_shp_subroot)
        shutil.rmtree(worker_finaloutput_subroot)
    except:
        pass
    
    # check local storage for temporary storage 
    # os.mkdir(worker_img_root)
    os.mkdir(worker_divided_img_subroot)
    os.mkdir(worker_output_shp_subroot)
    os.mkdir(worker_finaloutput_subroot)

    # ___________________________________________________________


    """  I AM TESTING ON UCONN HPC, DO NOT NEED TRANSFERING DATA
    # clean up the folder
    try:
        shutil.rmtree(worker_img_root)
        shutil.rmtree(worker_divided_img_root)
        shutil.rmtree(worker_output_shp_root)
        shutil.rmtree(worker_finaloutput_root)
    except:
        pass
    
    # check local storage for temporary storage 
    os.mkdir(worker_img_root)
    os.mkdir(worker_divided_img_root)
    os.mkdir(worker_output_shp_root)
    os.mkdir(worker_finaloutput_root)

    # move data from master to local worker 
    print("I am a worker with rank %d on %s moving data %s." % (rank, name, work))
    # cmd line
    cmd = "scp pi@master:%s pi@%s:%s"%(os.path.join(master_img_root,work),
                                                    name,
                                                    worker_img_root)
    # transfer the img
    os.system(cmd)
    """

    # ______________________run task______________________ 
    # (1) divide image
    x_resolution, y_resolution = divide.divide_image(input_img_path, 
                                                     worker_divided_img_subroot, 
                                                     crop_size)
    
    # (2) inference image
    inference.inference_image(worker_divided_img_subroot,
                              POLYGON_DIR,
                              weights_path,
                              worker_output_shp_subroot,
                              x_resolution, 
                              y_resolution)

    # (3) stitch shpfiles
    stitch.stitch_shapefile(worker_output_shp_subroot,
                            input_img_name,
                            worker_finaloutput_subroot)
    # ____________________________________________________

    pass


def loadwork(queue, filelist):
    for f in filelist:
        queue.put(f)


def master(comm):
    
    print ("Master starts working ...")

    num_procs = comm.Get_size()
    status = MPI.Status()
    
    # generate work queue on master node
    imgs_path = r"/pylon5/ps5fp1p/wez13005/local_dir/datasets/polygon/input_img/"
    imgs_path_list = [os.path.join(imgs_path,img_name) for img_name in os.listdir(imgs_path) if img_name.endswith('.tif')]
    wq = Work(imgs_path_list)

    # Seed the slaves, send one unit of work to each slave (rank)
    for rank in range(1, num_procs):
        work = wq.get_next()
        comm.send(work, dest=rank, tag=WORKTAG)
    
    # Loop over getting new work requests until there is no more work to be done
    while True:
        work = wq.get_next()
        if not work: break
    
        # Receive results from a slave
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        #process_result(result)

        # Send the slave a new work unit
        comm.send(work, dest=status.Get_source(), tag=WORKTAG)
    
    # No more work to be done, receive all outstanding results from slaves
    for rank in range(1, num_procs): 
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        #process_result(result)

    # Tell all the slaves to exit by sending an empty message with DIETAG
    for rank in range(1, num_procs):
        comm.send(0, dest=rank, tag=DIETAG)
    
    
def worker(comm):

    print ("Worker starts working ...")

    my_rank = comm.Get_rank()
    my_name = MPI.Get_processor_name()
    status = MPI.Status()

    while True:
        # Receive a message from the master
        work = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        # Check the tag of the received message
        if status.Get_tag() == DIETAG: break 

        # Do the work
        result = processing_img(my_rank,my_name,work)

        # Send the result back
        comm.send(result, dest=0, tag=0) 
    

def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    my_name = MPI.Get_processor_name()
    
    # main function
    comm.Barrier()
    start = MPI.Wtime()

    if my_rank == 0:
        master(comm)
    else:
        worker(comm)
    
    comm.Barrier()
    end = MPI.Wtime()
    print("Total time: ", end-start)


if __name__ == '__main__':
    main()
