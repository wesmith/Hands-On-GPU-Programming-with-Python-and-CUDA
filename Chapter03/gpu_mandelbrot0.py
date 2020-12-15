import matplotlib
# the following line will prevent the figure from popping up
matplotlib.use('Agg')  # WS Agg is the backend for writing a file, not for using show()
# WS: to show the image on screen, turn off above line, add plt.show() after plt.imshow()
import numpy as np
import pycuda.autoinit
from matplotlib import pyplot as plt
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from time import time

mandel_ker = ElementwiseKernel(
"pycuda::complex<float> *lattice, float *mandelbrot_graph, int max_iters, float upper_bound",
"""
mandelbrot_graph[i] = 1;

pycuda::complex<float> c = lattice[i]; 
pycuda::complex<float> z(0,0);

for (int j = 0; j < max_iters; j++)
    {
    
     z = z*z + c;
     
     if(abs(z) > upper_bound)
         {
          mandelbrot_graph[i] = 0;
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
    print('shape of lattice: {}'.format(mandelbrot_lattice.shape))  # WS mod

    # copy complex lattice to the GPU
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)

    # allocate an empty array on the GPU
    mandelbrot_graph_gpu = gpuarray.empty(shape=mandelbrot_lattice.shape, dtype=np.float32)

    mandel_ker(mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters),
               np.float32(upper_bound))
              
    mandelbrot_graph = mandelbrot_graph_gpu.get()
    
    return mandelbrot_graph


if __name__ == '__main__':

    siz = 512 * 8  # WS mod
    # WS: matplotlib doesn't know pixels: it uses dpi and size in inches to scale the saved image
    user_dpi = 100

    t1 = time()
    mandel = gpu_mandelbrot(siz, siz, -2, 2, -2, 2, 256, 2)  # WS mod added variable siz
    t2 = time()

    mandel_time = t2 - t1

    t1 = time()

    fig = plt.figure(1, frameon=False)  # WS mod
    # WS: setting size in inches is a critical step to preserve pixel size in saved image
    fig.set_size_inches(siz/user_dpi, siz/user_dpi)

    print('shape of Mandelbrot image: {}'.format(mandel.shape))  # WS mod
    plt.imshow(mandel) # WS mod, turn off exent to see pixel values on axes; extent=(-2, 2, -2, 2)
    #plt.show()  # WS addition: this shows image on screen if 'Agg' above turned off

    # WS: critical to include dpi here to preserve pixel size in saved image
    plt.savefig('Chapter03/mandelbrot_{}.png'.format(siz), bbox_inches='tight', dpi=user_dpi)

    t2 = time()

    plt.close(fig)  # WS mod: probably not needed here

    dump_time = t2 - t1

    print('It took {} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {} seconds to dump the image.'.format(dump_time))
