import numpy as np
import tensorflow as tf
from qml.ml import representations
from qml.aglaia import symm_funct
import time
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


mem_output = open("batches_memory.txt", 'a')
p = psutil.Process()

## ------------- ** Parameters for acsf ** -------------------------

rad_rs = np.arange(0, 10, 5)
ang_rs = np.arange(0.5, 10.5, 5)
theta_s = np.arange(0, 5, 5)
zeta = 8.0
eta = 4.0
radial_cutoff = 10.0
angular_cutoff = 10.0

## ------------- ** Making the descriptor ** ------------

batch_sizes = [int(arguments[1])]
samples = [int(arguments[2])]

for n_points in samples:

    for batch_size in batch_sizes:

        for i in range(1):

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
            sess_batches.close()

mem_output.write('{0} {1} {2}'.format(arguments[2], arguments[1], p.memory_info()[1]))
mem_output.write("\n")
