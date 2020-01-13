from pathlib import Path
import cv2
import numpy as np
import itertools

# img_folders = ["/Users/gabrielbirman/COS429/Final Project/Data/FSD/Original"] # FSD 
# img_folders = ["/Users/gabrielbirman/COS429/Final Project/Data/Pratheepan/Pratheepan_Dataset/FacePhoto", 
#             "/Users/gabrielbirman/COS429/Final Project/Data/Pratheepan/Pratheepan_Dataset/FamilyPhoto"] # Pratheepan Total
# img_folders = ["/Users/gabrielbirman/COS429/Final Project/Data/Pratheepan/Pratheepan_Dataset/FacePhoto"] # Pratheepan Face
img_folders = ["/Users/gabrielbirman/COS429/Final Project/Data/Pratheepan/Pratheepan_Dataset/FamilyPhoto"] # Pratheepan Family

N = 0 
b_mu = g_mu = r_mu = 0 # channel means 
b_ss = g_ss = r_ss = 0 # channel square sums 
for img_path in itertools.chain(*[Path(img_folder).iterdir() for img_folder in img_folders]):

    img = cv2.imread(str(img_path)).astype("uint16") # loads in BGR image 

    # get individual channels 
    b_chan = img[:,:,0]
    g_chan = img[:,:,1]
    r_chan = img[:,:,2]

    # update channel means 
    b_mu = (b_mu * N + np.mean(b_chan)) / (N + 1)
    g_mu = (g_mu * N + np.mean(g_chan)) / (N + 1)
    r_mu = (r_mu * N + np.mean(r_chan)) / (N + 1)

    # update (mean) squared sum 
    b_ss += np.mean(b_chan ** 2)
    g_ss += np.mean(g_chan ** 2)
    r_ss += np.mean(r_chan ** 2)

    N += 1 

# variance 
b_var = b_ss / N - b_mu**2
g_var = g_ss / N - g_mu**2
r_var = r_ss / N - r_mu**2

# standard deviation 
b_std = np.sqrt(b_var)
g_std = np.sqrt(g_var)
r_std = np.sqrt(r_var)

print(f"mu --- B: {b_mu:.1f}, G: {g_mu:.1f}, R: {r_mu:.1f}")
print(f"var --- B: {b_var:.1f}, G: {g_var:.1f}, R: {r_var:.1f}")
print(f"std --- B: {b_std:.1f}, G: {g_std:.1f}, R: {r_std:.1f}")



