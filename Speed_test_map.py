import numpy as np
import tensorflow as tf
from qml.ml import representations
import time
import symm_funct_map
import psutil
import sys

## ------------ ** Getting the arguments ** -----------

arguments = sys.argv

## ------------- ** Loading the data ** ---------------

data = np.load("xyz_cnisopent.npz")
xyz = data["arr_0"]
zs = data["arr_1"]

n_samples = xyz.shape[0]
max_n_atoms = xyz.shape[1]

mbtypes = representations.get_slatm_mbtypes([zs[i] for i in range(zs.shape[0])])

elements = []
element_pairs = []

# Splitting the one and two body interactions in mbtypes
for item in mbtypes:
    if len(item) == 1:
        elements.append(item[0])
    if len(item) == 2:
        element_pairs.append(list(item))
    if len(item) == 3:
        break


mem_output = open("map_memory.txt", 'a')
p = psutil.Process()

## ------------- ** Parameters for acsf ** -------------------------

rad_rs = np.arange(0,10, 5)
ang_rs = np.arange(0.5, 10.5, 5)
theta_s = np.arange(0, 5, 5)
zeta = 8.0
eta = 4.0
radial_cutoff = 10.0
angular_cutoff = 10.0

## ------------- ** Making the descriptor  ** ------------

samples = [int(arguments[1])]

def generate_descriptor(batch_xyz, batch_zs):
    
    descriptor = symm_funct_map.generate_parkhill_acsf_1(xyzs=batch_xyz, Zs=batch_zs, elements=elements,
                                             element_pairs=element_pairs,
                                             radial_cutoff=radial_cutoff, angular_cutoff=angular_cutoff,
                                             radial_rs=rad_rs, angular_rs=ang_rs,
                                             theta_s=theta_s,
                                             eta=eta, zeta=zeta)

    return descriptor, batch_zs

for n_points in samples:

    for i in range(1):
        map_start_time = time.time()

        tf.reset_default_graph()

        with tf.name_scope("Data"):
            zs_tf = tf.placeholder(shape=[n_points, max_n_atoms], dtype=tf.int32, name="zs")
            xyz_tf = tf.placeholder(shape=[n_points, max_n_atoms, 3], dtype=tf.float32, name="xyz")

            dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
            dataset = dataset.map(generate_descriptor)
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            batch_descriptor, batch_zs = iterator.get_next()


        sess_map = tf.Session()
        sess_map.run(tf.global_variables_initializer())
        sess_map.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyz[:n_points], zs_tf: zs[:n_points]})

        descriptor_map_slices=[]

        batch_counter = 0
        while True:
            try:
                descriptor_np = sess_map.run(batch_descriptor)
                descriptor_map_slices.append(descriptor_np)
                batch_counter += 1
            except tf.errors.OutOfRangeError:
                print("Finished iteration %s with %s data points." % (str(i), str(n_points)))
                break

        descriptor_map_conc = np.asarray(descriptor_map_slices)

        map_end_time = time.time()

        final_map_time = map_end_time - map_start_time

        sess_map.close()

mem_output.write('{0} {1}'.format(arguments[1], p.memory_info()[1]))
mem_output.write("\n")