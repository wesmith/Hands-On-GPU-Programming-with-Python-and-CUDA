# Conway's game of life in Python / CUDA C
# written by Brian Tuomanen for "Hands on GPU Programming with Python and CUDA"

# WS 12/17/20: modified to conway_gpu_opencv.py to get animation running in openCV: works great

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time

ker = SourceModule("""
#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )

#define _WIDTH  ( blockDim.x * gridDim.x )
#define _HEIGHT ( blockDim.y * gridDim.y  )

#define _XM(x)  ( (x + _WIDTH) % _WIDTH )
#define _YM(y)  ( (y + _HEIGHT) % _HEIGHT )

#define _INDEX(x,y)  ( _XM(x)  + _YM(y) * _WIDTH )

// return the number of living neighbors for a given cell                
__device__ int nbrs(int x, int y, int * in)
{
    return ( in[_INDEX(x-1, y+1)] + in[_INDEX(x-1, y)] + in[_INDEX(x-1, y-1)] \
           + in[_INDEX(x,   y+1)]                      + in[_INDEX(x,   y-1)] \
           + in[_INDEX(x+1, y+1)] + in[_INDEX(x+1, y)] + in[_INDEX(x+1, y-1)] );
}

__global__ void conway_ker(int * lattice_out, int * lattice)
{
    // x, y are the appropriate values for the cell covered by this thread
    int x = _X, y = _Y;

    // count the number of neighbors around the current cell
    int n = nbrs(x, y, lattice);
                    
    // if the current cell is alive, then determine if it lives or dies for the next generation.
    if (lattice[_INDEX(x,y)] == 1)
        switch(n)
        {
            // if the cell is alive: it remains alive only if it has 2 or 3 neighbors.
            case 2:
            case 3: lattice_out[_INDEX(x,y)] = 1;
                    break;
            default: lattice_out[_INDEX(x,y)] = 0;                   
        }
    else if(lattice[_INDEX(x,y)] == 0 )
        switch(n)
        {
            // a dead cell comes to life only if it has 3 neighbors that are alive.
            case 3: lattice_out[_INDEX(x,y)] = 1;
                    break;
            default: lattice_out[_INDEX(x,y)] = 0;         
        }   
}
""")

conway_ker = ker.get_function("conway_ker")

def update_gpu(newLattice_gpu, lattice_gpu, N):  # WS rewrite for openCV display

    # WS mod: for python3, had to cast N/32 to int or else 'no registered converter...' error
    conway_ker(newLattice_gpu, lattice_gpu, grid=(int(N/32),int(N/32),1), block=(32,32,1))

    lattice_gpu[:] = newLattice_gpu[:]

    return newLattice_gpu.get()


if __name__ == '__main__':
    # set lattice size
    N = 128 * 5      # > 768x768 starts to slow down a little: bottlenecked by transfer of gpu mem to cpu?
    window_scale = 1 # scaling the window > 1 slows down the display a little for large N
    pval = 0.25 # WS mod: probability of 1's (0.25 is nominal value)

    lattice     = np.int32(np.random.choice([1,0], N*N, p=[pval, 1-pval]).reshape(N, N))
    lattice_gpu = gpuarray.to_gpu(lattice)

    newLattice_gpu = gpuarray.empty_like(lattice_gpu) 

    txt = 'Game of Life {} x {} (press q to quit)'.format(N, N)
    cv2.namedWindow(txt, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(txt, window_scale*N, window_scale*N)

    # WS mod: flash up first frame
    dd = (255 * lattice).astype(np.uint8)
    cv2.imshow(txt, dd)
    cv2.moveWindow(txt, 0, 0)
    print('Starting frame: press any key to continue, then "q" to finally quit')
    cv2.waitKey(0)

    while True:  # couldn't get 'for' loop to work: 'while' loop works with waitKey

        dd = update_gpu(newLattice_gpu, lattice_gpu, N)
        dd = (255 * dd).astype(np.uint8)

        cv2.imshow(txt, dd)
        cv2.moveWindow(txt, 0, 0)
        #time.sleep(0.001)
        k = cv2.waitKey(1)  # don't use 0: it freezes

        if k == ord('q'): 
            break

    cv2.destroyAllWindows()
