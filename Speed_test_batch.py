import numpy as np
import tensorflow as tf
from qml.ml import representations
from qml.aglaia import symm_funct
import os
import time
import psutil
import sys

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

output = open("batches_results.txt", 'w')
output.write("\n These results were generated with the following data: ")
output.write("n_atoms: %s, elements: %s" % (str(max_n_atoms), str(elements)))
output.write("\n")

## ------------- ** Parameters for acsf ** -------------------------

rad_rs = np.arange(0, 10, 5)
ang_rs = np.arange(0.5, 10.5, 5)
theta_s = np.arange(0, 5, 5)
zeta = 8.0
eta = 4.0
radial_cutoff = 10.0
angular_cutoff = 10.0

## ------------- ** Making the descriptor ** ------------

batch_sizes = [1, 5, 50, 200, 400]
samples = [400, 800, 1500, 2500, 3000]


full_batch_sizes = []
times = []
full_n_samples = []

for n_points in samples:

    for batch_size in batch_sizes:

        for i in range(5):

            batch_start_time = time.time()

            tf.reset_default_graph()

            with tf.name_scope("Inputs"):
                zs_tf = tf.placeholder(shape=[n_points, max_n_atoms], dtype=tf.int32, name="zs")
                xyz_tf = tf.placeholder(shape=[n_points, max_n_atoms, 3], dtype=tf.float32, name="xyz")

                dataset = tf.data.Dataset.from_tensor_slices((xyz_tf, zs_tf))
                dataset = dataset.batch(batch_size)
                iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
                batch_xyz, batch_zs = iterator.get_next()

            descriptor = symm_funct.generate_parkhill_acsf(xyzs=batch_xyz, Zs=batch_zs, elements=elements, element_pairs=element_pairs,
                                               radial_cutoff=radial_cutoff, angular_cutoff=angular_cutoff,
                                               radial_rs=rad_rs, angular_rs=ang_rs, theta_s=theta_s,
                                               eta=eta, zeta=zeta)

            sess_batches = tf.Session()
            sess_batches.run(tf.global_variables_initializer())
            sess_batches.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyz[:n_points], zs_tf: zs[:n_points]})

            descriptor_slices=[]

            batch_counter = 0
            while True:
                try:
                    descriptor_np = sess_batches.run(descriptor)
                    descriptor_slices.append(descriptor_np)
                    batch_counter += 1
                except tf.errors.OutOfRangeError:
                    print("Finished batch %s iteration %s. \n" % (str(batch_size), str(i)))
                    break

            descriptor_conc = np.concatenate(descriptor_slices, axis=0)

            batch_end_time = time.time()

            final_time = batch_end_time - batch_start_time
            times.append(final_time)
            full_batch_sizes.append(batch_size)
            full_n_samples.append(n_points)

            output.write("The time taken for the descriptor in batches of %s is: %s \n" % (str(batch_size), str(final_time)))

            sess_batches.close()

    output.write("The shape of the descriptor is %s" % (str(descriptor_conc.shape)))

times = np.asarray(times)
full_batch_sizes = np.asarray(full_batch_sizes)
full_n_samples = np.asarray(full_n_samples)

np.savez("batches_results.npz", full_n_samples, full_batch_sizes, times)
