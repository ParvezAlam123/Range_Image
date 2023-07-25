import numpy as np 
import open3d as o3d 
import struct 
import math
import torch
import matplotlib.pyplot as plt
from PIL import Image




def ProjectPCimg2SphericalRing(PC, H_input = 64, W_input = 1800):
    
    num_points, dim = PC.shape

    degree2radian = math.pi / 180
    nLines = H_input    
    AzimuthResolution = 360.0 / W_input # degree
    FODown = -(-24.8) # take absolute value
    FOUp = 2.0
    FOV = FODown+FOUp
    
    
    range_img = np.zeros((H_input+1, W_input+1), dtype=np.float32)
    for i in range(num_points):
        x = PC[i][0]
        y = PC[i][1]
        z = PC[i][2]
        r = math.sqrt(x**2+y**2+z**2)
        pitch = math.asin(z / r) * (180/np.pi)
        yaw = math.atan2(y, x)   * (180/np.pi)
        u = int(64 * ((FOUp - pitch)/FOV)) 
        v = int(1800 * ((yaw+180)/360)) 
        
        range_img[u][v] = r
  
    return range_img






path = "/media/parvez_alam/Expansion/Kitti/Odometry/data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin" 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

list_pcd = [] 
with open(path, "rb") as f:
    size_float = 4
    byte = f.read(size_float * 4)
    while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
points = np.asarray(list_pcd)



PC_project_final = ProjectPCimg2SphericalRing(points) 



img_plot = plt.imshow(PC_project_final)
plt.show()
 
