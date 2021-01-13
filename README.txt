*****READ ME*****

Important Notes:

-This program is used to test the performance of a seperable filter convolution on the mandlebrot set.

-The Mandlebrot set image creation has it's own computation within the source code that is not considered in the performance evaluations.

How to Use/Process Explained:

-Upon starting the program you will be prompted to enter the standard deviation value for generating the gaussian distribution values. These values are then used in the computation to blur the mandelbrot image.
-During the computation process the program will go through 2 phases: a non-tiled computation and a tiled. In each respective phase the computation will go through 10 iterations of calculating a median from a data set of 10 convolution computations. The Median times for each phase are then displayed
-by default the resolution at which the madelbrot image is set is 1024 x 768 (This is specifically set because it can be used with the largest tile size: 1024)

-If you want to change the resolution size you will also have to change the Tile Size(TS)
	-below is a small chart I used for which resolutions align to TS (Each tile size should be a multiple of 2)
	-*any resolution can be set as long as it is divisible by the TS

Resolution    |    Tile Size    |
--------------|-----------------|
800 x 600     |    2 - 8        |
--------------|-----------------|
1024 x 768    |    2 - 1024     |
--------------|-----------------|
1280 x 960    |    2 - 32       |
--------------|-----------------|
1440 x 1563   |    2 - 8        |
--------------|-----------------|
2048 x 1024   |    2 - 1024     |
--------------|-----------------|


Know Bugs:
-When Tile set below 128 the full image does not render the blur
-If program ran multiple times, clear the generated image files and cvs files otherwise Tiled phase will not run
