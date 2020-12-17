# ws_gpu_mandelbrot_color.py
# WESmith 12/16/20 modified from gpu_mandelbrot0.py to produce a color version with openCV
# note: prettier color pictures may result from using various lookup tables in matlab on a
# single channel with the grayscale mod in mandel_ker, instead of using a 3-channel approach
# in openCV

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from time import time
import cv2  # WS mod use openCV to display and manipulate color output

mandel_ker = ElementwiseKernel(
"pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
"""
//mandelbrot_graph[i] = max_iters - 1;  // WS mod: max_iters should be no larger than 256
mandelbrot_graph[i] = 0;  // WS mod: show just the Mandelbrot borders: stable = 0, most unstable = 0

pycuda::complex<float> c = lattice[i]; 
pycuda::complex<float> z(0,0);

for (int j = 0; j < max_iters; j++)
    {
    
     z = z*z + c;
     
     if(abs(z) > upper_bound)
         {
          mandelbrot_graph[i] = j;   // WS mod for grayscale to reflect convergence
          break;
         }

    }
         
""",
"mandel_ker")

def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):

    # we set up our complex lattice as such
    real_vals = np.matrix(np.linspace(real_low,  real_high, width), dtype=np.complex64)
    imag_vals = np.matrix(np.linspace(imag_high, imag_low, height), dtype=np.complex64) * 1j
    mandelbrot_lattice = np.array(real_vals + imag_vals.transpose(), dtype=np.complex64)
    #print('shape of lattice: {}'.format(mandelbrot_lattice.shape))  # WS mod

    # copy complex lattice to the GPU
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)

    # allocate an empty array on the GPU
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)

    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters),
               np.float32(upper_bound))
              
    mandelbrot_graph = mandelbrot_graph_gpu.get()
    
    return mandelbrot_graph


def linear_interp(a, dmin, dmax):
    # linear interp of numpy array a to desired dmin, dmax
    amin = a.min()
    slope = (dmax - dmin) / (a.max() - amin)
    return slope * (a - amin) + dmin


if __name__ == '__main__':

    siz       = 512 * 2  # WS mod
    max_iters = 256      # WS mod
    # WS mod: three_channel: True  use 3-channel color display, with slightly different upper 
    #                              bounds per channel
    #                       vFalse use an openCV color map to color a single channel
    three_channel = False
    real_min, real_max, imag_min, imag_max = (-2.2, 0.8, -1.5, 1.5) # bounds of complex plane

    if three_channel:  
        #upper_bound = [1.0, 1.5, 2.0]  # WS mod: ranges of upper bounds
        upper_bound = [2.0, 1.6, 1.3]
        #upper_bound = [4.0, 8.0, 16.0]
        img = np.zeros((siz, siz, 3), dtype=np.uint8)
        txt = 'three_channels'
    else:
        upper_bound = [2.0]  # nominal single-channel value
        img = np.zeros((siz, siz), dtype=np.uint8)
        txt = 'single channel'
    
    for i, k in enumerate(upper_bound):  # WS mod to generate color channels

        t1 = time()
        mandel = gpu_mandelbrot(siz, siz, real_min, real_max, imag_min, imag_max, max_iters, k)
        t2 = time()
        
        mandel = np.log10(mandel + 1)  # stretch values for a better dynamic range
        mandel = linear_interp(mandel, 0, 255)

        if three_channel:
            img[:, :, i] = mandel.astype(np.uint8)
        else:
            img = mandel.astype(np.uint8)

        mandel_time = t2 - t1
        print('{} seconds to calculate channel {}.'.format(mandel_time, i))

    print('shape of Mandelbrot image: {}'.format(img.shape))  # WS mod

    if not three_channel:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    while True:

        cv2.imshow('Mandelbrot_{}'.format(txt), img)
        cv2.moveWindow('Mandelbrot_{}'.format(txt), 0, 0)

        k = cv2.waitKey(0)

        if k == ord('q'): 
            break
        elif k == ord('s'):
            print('saving image to disk')
            cv2.imwrite('Chapter03/Mandelbrot_color_{}_{}.png'.format(txt, siz), img)
            break
    
    cv2.destroyAllWindows()


   
