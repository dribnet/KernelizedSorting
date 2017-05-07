## Kernelized Sorting for Image Layouting with color features
## param[in] psize sampling size of image
## param[in] ino width of the layout area
## param[in] jno height of the layout area
## param[in] iname file name for the images

from kernelized_sorting_color import KS
from utils import lab
import matplotlib.pylab as mp
from PIL import Image
import numpy
import pdb
import glob

ino = 17
jno = 17
ysize = 128
xsize = 224
iname = 'images/img'

imgdata = []
data = []
counter = 0
images = glob.glob("/develop/data/region-flags/png/close/*.png")

# for i in range(ino):
#     for j in range(jno):
for fname in images:
    # counter = counter + 1
    # fname = iname + str(counter) + '.jpg'
    im = Image.open(fname).convert('RGB')
    aim = numpy.asarray(im)
    print(aim.shape)
    [M,N,L] = aim.shape
    mno = int(M/ysize)
    nno = int(N/xsize)
    aim = aim[0:ysize*mno:mno,0:xsize*nno:nno,:]
    data.append(aim.flatten())
    daim = numpy.double(aim)/255.0
    # convert from RGB to Lab
    daimlab = lab(daim)
    imgdata.append(daimlab.flatten())
        
data = numpy.array(data)
imgdata = numpy.array(imgdata)
griddata = numpy.zeros((2,ino*jno))
griddata[0,] = numpy.kron(range(1,ino+1),numpy.ones((1,jno)))
griddata[1,] = numpy.tile(range(1,jno+1),(1,ino))

# do kernelized sorting procedure
PI = KS(imgdata,griddata.T)
i_sorting = PI.argmax(axis=1)
imgdata_sorted = data[i_sorting,]
irange = range(0,ysize*ino,ysize)
jrange = range(0,xsize*jno,xsize)
patching = numpy.zeros((ino*ysize, jno*xsize, 3))
for i in range(ino):
    for j in range(jno):
        patching[irange[i]:irange[i]+ysize,jrange[j]:jrange[j]+xsize,:] = numpy.reshape(imgdata_sorted[(i)*jno+j,], [ysize,xsize,3]);

im = Image.fromarray(patching.astype(numpy.uint8))
# im.show()
im.save("output_color.png")
