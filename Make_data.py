import glob
from qml.aglaia.aglaia import MRMP
import numpy as np

filenames = glob.glob("/Volumes/Transcend/data_sets/CN_isobutane_model/geoms_2/training/*.xyz")[:10000]

estimator = MRMP()
estimator.generate_compounds(filenames)

xyz = []
zs = []

for item in estimator.compounds:
    xyz.append(item.coordinates)
    zs.append(item.nuclear_charges)

xyz = np.asarray(xyz)
zs = np.asarray(zs)
print(xyz.shape, zs.shape)

np.savez("xyz_cnisopent.npz", xyz, zs)

