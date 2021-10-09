import os
import shutil

path = './test_outputs'
if not os.path.exists(path):
    os.makedirs(path)

base = './mvs_training/dtu_test'
scans = os.listdir(base)
scans_num = [scan[4:] for scan in scans]
print(scans_num)

for i in range(len(scans)):
    p = os.path.join(base, scans[i], outputs, 'points_mvsnet')
    ls = os.listdir(p)
    n = None
    for l in ls :
        if l.startswith('consistencyCheck-20210508'):
            n = l
    p = os.path.join(p, n, 'final3d_model.ply')
    shutil.copy(p, os.path.join(path, 'mvsnet' + scans_num[i] +'_l3.ply'))