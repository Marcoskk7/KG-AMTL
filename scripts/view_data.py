import scipy.io as sio
data = sio.loadmat("/home/marcos/KG-AMTL/data/CWRU/48k Drive End Bearing Fault Data/Ball/0007/B007_0.mat")
print(data.keys())
print(data)