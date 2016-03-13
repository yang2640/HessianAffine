Installation: 
(1) This project depends on the vlfeat library, install vlfeat, especially vlfeat.so
(2) Modify on my Makefile (paths) to compile
(3) refer to the command usage for help


Hessian Affine feature detector + sift descriptor.

Available options
-----------------
-h, --help
    useage: ./sift -estimateaffineshape -estimateorientation
    -peakthreshold=0.001 imgpath outpath

--doubleimg
    bool option: increase image resolution

--estimateaffineshape
    bool option: estimate affineshape

--estimateorientation
    bool option: estimate orientation

--verbose
    bool option: output more debug info

--peakthreshold <value>
    peak threhold value for hessian, default value is 0.001

Return codes
-----------------
    -1    Error
    0     Success
