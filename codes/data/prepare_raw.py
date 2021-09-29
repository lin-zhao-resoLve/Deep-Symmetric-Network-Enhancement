import os
import rawpy
import imageio
def store_dataset(dir):
    images = []
    all_path = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
                # img = Image.open(path).convert('RGB')
                # images.append(img)
            all_path.append(path)
            names.append(fname)

    return all_path, names

file1 = '/home/lin-zhao/Sony/short1'
filename1, name1s = store_dataset(file1)
for i in range(len(filename1)):
    name = os.path.split(name1s[i])[-1].replace('.ARW', '.png')
    if os.path.exists('/home/lin-zhao/Sony/LQ/'+name):
        continue
    print(i)
    low_raw = rawpy.imread(filename1[i])
    im = low_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
    imageio.imwrite('/home/lin-zhao/Sony/LQ/'+name, im)