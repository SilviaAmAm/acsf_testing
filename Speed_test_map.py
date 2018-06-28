import numpy as np
import tensorflow as tf
from qml.ml import representations
import os
import time
import symm_funct_map
import psutil

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

output = open("map_results.txt", 'w')
output.write("These results were generated with the following data:")
output.write("n_atoms: %s, elements: %s" % (str(max_n_atoms), str(elements)))
output.write("\n")

mem_output = open("map_memory.txt", 'a')
p = psutil.Process()
mem_output.write("The current process is: %s. \n" % (str(p.name)))


## ------------- ** Parameters for acsf ** -------------------------

rad_rs = np.arange(0,10, 5)
ang_rs = np.arange(0.5, 10.5, 5)
theta_s = np.arange(0, 5, 5)
zeta = 8.0
eta = 4.0
radial_cutoff = 10.0
angular_cutoff = 10.0

## ------------- ** Making the descriptor  ** ------------

# run_metadata = tf.RunMetadata()
# options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# samples = [100, 200, 400, 800, 3000]
samples = [10000]

full_batch_sizes = []
times = []
full_n_samples = []

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
        # sess_map.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyz[:n_points], zs_tf: zs[:n_points]}, options=options, run_metadata=run_metadata)
        sess_map.run(iterator.make_initializer(dataset), feed_dict={xyz_tf: xyz[:n_points], zs_tf: zs[:n_points]})

        descriptor_map_slices=[]

        # path = "tensorboard/map/n_samples_" + str(n_points) + "_iter_" + str(i)
        #
        # if not os.path.exists(path):
        #     os.makedirs(path)
        #
        # summary_writer = tf.summary.FileWriter(logdir=path, graph=sess_map.graph)

        batch_counter = 0
        while True:
            try:
                descriptor_np = sess_map.run(batch_descriptor)
                # summary_writer.add_run_metadata(run_metadata=run_metadata, tag="batch %s" % batch_counter, global_step=None)
                descriptor_map_slices.append(descriptor_np)
                batch_counter += 1
            except tf.errors.OutOfRangeError:
                print("Finished iteration %s with %s data points." % (str(i), str(n_points)))
                break

        descriptor_map_conc = np.asarray(descriptor_map_slices)

        map_end_time = time.time()

        final_map_time = map_end_time - map_start_time

        times.append(final_map_time)
        full_n_samples.append(n_points)

        output.write("\n The time taken for the descriptor with map is: %s (using %s data points) \n" % (str(final_map_time), str(n_points)))

        sess_map.close()

    output.write("The shape of the descriptor is %s. \n" % (str(descriptor_map_conc.shape)))

mem_output.write(str(p.memory_info()))
mem_output.write("\n")

times = np.asarray(times)
full_n_samples = np.asarray(full_n_samples)

np.savez("map_results.npz", full_n_samples, times)