// Data Structures and Algorithms II : Intro to AMP and benchmarking exercise
// Ruth Falconer  <r.falconer@abertay.ac.uk>
// Adapted from C++ AMP book http://ampbook.codeplex.com/license.

/*************************To do in lab ************************************/
//Change size of the vector/array
//Compare Debug versus Release modes
//Add more work to the loop until the GPU is faster than CPU
//Compare double versus ints via templating  

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
//needed for GPU programming
#include <amp.h>
#include <amp_math.h>
#include <amp_graphics.h>
#include <cassert>
#include <iomanip>
#include <time.h>
#include <string>
#include <array>
#include <assert.h>

//for gaussian dist.
#define _USE_MATH_DEFINES
#include <math.h>

#define SIZE 1<<25 // same as 2^24

// Need to access the concurrency libraries 
using namespace concurrency;

// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::cout;
using std::endl;
using std::ofstream;

using std::chrono::duration_cast;
// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_clock;

// The size of the image to generate.
const int WIDTH =  2048 * 2;
const int HEIGHT = 1024 * 2;
const int TS = 256;


// The number of times to iterate before we assume that a point isn't in the
// Mandelbrot set.
// (You may need to turn this up if you zoom further into the set.)
const int MAX_ITERATIONS = 800;

// The image data.
// Each pixel is represented as 0xRRGGBB.
uint32_t image[HEIGHT][WIDTH];

//need for the GPU
using namespace concurrency;
using namespace concurrency::graphics;

void report_accelerator(const accelerator a)
{
	const std::wstring bs[2] = { L"false", L"true" };
	std::wcout << ": " << a.description << " "
		<< endl << "       device_path                       = " << a.device_path
		<< endl << "       dedicated_memory                  = " << std::setprecision(4) << float(a.dedicated_memory) / (1024.0f * 1024.0f) << " Mb"
		<< endl << "       has_display                       = " << bs[a.has_display]
		<< endl << "       is_debug                          = " << bs[a.is_debug]
		<< endl << "       is_emulated                       = " << bs[a.is_emulated]
		<< endl << "       supports_double_precision         = " << bs[a.supports_double_precision]
		<< endl << "       supports_limited_double_precision = " << bs[a.supports_limited_double_precision]
		<< endl;
}
// List and select the accelerator to use
void list_accelerators()
{
	//get all accelerators available to us and store in a vector so we can extract details
	std::vector<accelerator> accls = accelerator::get_all();

	// iterates over all accelerators and print characteristics
	for (size_t i = 0; i < accls.size(); i++)
	{
		accelerator a = accls[i];
		report_accelerator(a);

	}

	//Use default accelerator
	accelerator a = accelerator(accelerator::default_accelerator);
	std::wcout << " default acc = " << a.description << endl;
} // list_accelerators

// query if AMP accelerator exists on hardware
void query_AMP_support()
{
	std::vector<accelerator> accls = accelerator::get_all();
	if (accls.empty())
	{
		cout << "No accelerators found that are compatible with C++ AMP" << std::endl;
	}
	else
	{
		cout << "Accelerators found that are compatible with C++ AMP" << std::endl;
		list_accelerators();
	}
} // query_AMP_support

// using our own Complex number structure and definitions as the Complex type is not available
//in the Concurrency namespace
struct Complex1
{
	float x;
	float y;
};

Complex1 c_add(Complex1 c1, Complex1 c2) restrict(cpu, amp) // restrict keyword - able to execute this function on the GPU and CPU
{
	Complex1 tmp;
	float a = c1.x;
	float b = c1.y;
	float c = c2.x;
	float d = c2.y;
	tmp.x = a + c;
	tmp.y = b + d;
	return tmp;
}// c_add

float c_abs(Complex1 c) restrict(cpu, amp)
{
	return concurrency::fast_math::sqrt(c.x * c.x + c.y * c.y);
}// c_abs

Complex1 c_mul(Complex1 c1, Complex1 c2) restrict(cpu, amp)
{
	Complex1 tmp;
	float a = c1.x;
	float b = c1.y;
	float c = c2.x;
	float d = c2.y;
	tmp.x = a * c - b * d;
	tmp.y = b * c + a * d;
	return tmp;
}// c_mu

// Write the image to a TGA file with the given name.
// Format specification: http://www.gamers.org/dEngine/quake3/TGA.txt
void write_tga(const char* filename)
{
	ofstream outfile(filename, ofstream::binary);

	uint8_t header[18] = {
		0, // no image ID
		0, // no colour map
		2, // uncompressed 24-bit image
		0, 0, 0, 0, 0, // empty colour map specification
		0, 0, // X origin
		0, 0, // Y origin
		WIDTH & 0xFF, (WIDTH >> 8) & 0xFF, // width
		HEIGHT & 0xFF, (HEIGHT >> 8) & 0xFF, // height
		24, // bits per pixel
		0, // image descriptor
	};
	outfile.write((const char*)header, 18);

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			uint8_t pixel[3] = {
				image[y][x] & 0xFF, // blue channel
				(image[y][x] >> 8) & 0xFF, // green channel
				(image[y][x] >> 16) & 0xFF, // red channel
			};
			outfile.write((const char*)pixel, 3);
		}
	}

	outfile.close();
	if (!outfile)
	{
		// An error has occurred at some point since we opened the file.
		cout << "Error writing to " << filename << endl;
		exit(1);
	}
}


// Render the Mandelbrot set into the image array.
// The parameters specify the region on the complex plane to plot.
template <class T>
void compute_mandelbrot(T left, T right, T top, T bottom)
{
	//reference to the globally declared image data
	uint32_t* pImage = &(image[0][0]);


	//create array view object to encapsulate the image data (will be available to CPU and GPU)
	concurrency::array_view<uint32_t, 2> a(HEIGHT, WIDTH, pImage);

	//discard because data transfer between GPU and CPU wont be needed, all calc done on GPU
	a.discard_data();


	//parallel for each with lambda expression to run the mandlebrot set computation
	//create 640*480 threads to invoke the kernel
	try
	{
		concurrency::parallel_for_each(a.extent, [=](concurrency::index<2> idx) restrict(amp)
			{
				//because of the index being a 2d plane of thread id's
				// the y is mapped to the height of this plane and the 
				// x is mapped to the length
				//tip: index is [height][width]
				int y = idx[0];
				int x = idx[1];

				// Work out the point in the complex plane that
				// corresponds to this pixel in the output image.
				Complex1 c;
				c.x = left + (x * (right - left) / WIDTH);
				c.y = top + (y * (bottom - top) / HEIGHT);

				// Start off z at (0, 0).
				Complex1 z;
				z.x = 0.0;
				z.y = 0.0;

				// Iterate z = z^2 + c until z moves more than 2 units
				// away from (0, 0), or we've iterated too many times.
				int iterations = 0;
				while (c_abs(z) < 2.0 && iterations < MAX_ITERATIONS)
				{
					z = c_add(c_mul(z, z), c);

					++iterations;
				}

				if (iterations == MAX_ITERATIONS)
				{
					// z didn't escape from the circle.
					// This point is in the Mandelbrot set.
					a[y][x] = 0x000000; // black
				}
				else if (iterations < MAX_ITERATIONS / 100)
				{
					a[y][x] = 0x001540; // blue
				}
				else if (iterations < MAX_ITERATIONS / 50)
				{
					a[y][x] = 0x001c57; // blue
				}
				else if (iterations < MAX_ITERATIONS / 25)
				{
					a[y][x] = 0x00316e; // blue
				}
				else if (iterations < MAX_ITERATIONS / 15)
				{
					a[y][x] = 0xee204d; // red
				}
				else
				{
					// z escaped within less than MAX_ITERATIONS
					// iterations. This point isn't in the set.
					a[y][x] = 0xFFFFFF; // white
				}
			});
		//sync back written data
		a.synchronize();

	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}

}

template <class T>
void compute_mandelbrot_tile(T left, T right, T top, T bottom)
{
	static const int TS = 16;

	//reference to the globally declared image data
	uint32_t* pImage = &(image[0][0]);

	//create array view object to encapsulate the image data (will be available to CPU and GPU)
	concurrency::array_view<uint32_t, 2> a(HEIGHT, WIDTH, pImage);

	//discard because data transfer between GPU and CPU wont be needed, all calc done on GPU
	a.discard_data();

	//parallel for each with lambda expression to run the mandlebrot set computation
	//create 640*480 threads to invoke the kernel
	try
	{
		concurrency::parallel_for_each(a.extent.tile<TS, TS>(), [=](concurrency::tiled_index<TS,TS> t_idx) restrict(amp)
			{
				//because of the index being a 2d plane of thread id's
				// the y is mapped to the height of this plane and the 
				// x is mapped to the length
				//tip: index is [height][width]

				//access the index location from the global memory rather than the local 
				//because info is not shared between the threads so can not be placed in tile_static locations
				int y = t_idx.global[0];
				int x = t_idx.global[1];

				// Work out the point in the complex plane that
				// corresponds to this pixel in the output image.
				Complex1 c;
				c.x = left + (x * (right - left) / WIDTH);
				c.y = top + (y * (bottom - top) / HEIGHT);

				// Start off z at (0, 0).
				Complex1 z;
				z.x = 0.0;
				z.y = 0.0;

				// Iterate z = z^2 + c until z moves more than 2 units
				// away from (0, 0), or we've iterated too many times.
				int iterations = 0;
				while (c_abs(z) < 2.0 && iterations < MAX_ITERATIONS)
				{
					z = c_add(c_mul(z, z), c);

					++iterations;
				}

				if (iterations == MAX_ITERATIONS)
				{
					// z didn't escape from the circle.
					// This point is in the Mandelbrot set.
					a[y][x] = 0x000000; // black
				}
				else if (iterations < MAX_ITERATIONS / 100)
				{
					a[y][x] = 0x001540; // blue
				}
				else if (iterations < MAX_ITERATIONS / 50)
				{
					a[y][x] = 0x001c57; // blue
				}
				else if (iterations < MAX_ITERATIONS / 25)
				{
					a[y][x] = 0x00316e; // blue
				}
				else if (iterations < MAX_ITERATIONS / 15)
				{
					a[y][x] = 0xee204d; // red
				}
				else
				{
					// z escaped within less than MAX_ITERATIONS
					// iterations. This point isn't in the set.
					a[y][x] = 0xFFFFFF; // white
				}
			});
		//sync back written data
		a.synchronize();
	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}

}

// Function to create Gaussian filter 
void FilterCreation(float GKernel[][5], float sigma)
{
	// intialising standard deviation to 2.28 
	//sigma = 2.28;
	float r, s = 2.0 * sigma * sigma;

	// sum is for normalization 
	float sum = 0.0;

	// generating 5x5 kernel 
	for (int x = -2; x <= 2; x++) {
		for (int y = -2; y <= 2; y++) {
			r = sqrt(x * x + y * y);
			GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
			sum += GKernel[x + 2][y + 2];
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			GKernel[i][j] /= sum;
}

//will be used to flatten the filter kernel 2D matrix into a 1D vector
template <size_t rows, size_t cols>
std::vector<float> flatten(float (&a)[rows][cols])
{

	std::vector<float> temp;

	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			temp.push_back(a[r][c]);
		}
	}

	return temp;
}

//compute madlebrot and apply seperable filter convolution
template <class T>
void mandelbrot_convolution_sf(T sigma)
{

	//create necessary variables for filter 
	size_t radius = 5;
	size_t filter_size = radius * 2 + 1;
	std::vector<float> v_filter(radius * 2 + 1);

	//get gaussian distributed filter data 
	float GKernel[5][5];
	FilterCreation(GKernel, sigma);

	// initialize filter values
	//must flatten the GKernel from 2D array into 1D vector
	v_filter = flatten<5,5>(GKernel);

	//reference to the globally declared image data
	uint32_t* pImage = &(image[0][0]);

	//create array object to encapsulate the filter data
	extent<1> ek(v_filter.size());
	array<float, 1> a_filter(ek, v_filter.begin());

	//arrays for use of applying the filters.(buffer is for the first pass through of the seperable filter)
	array_view<uint32_t, 2>  a_img(HEIGHT, WIDTH, pImage), a_img_buffer(HEIGHT, WIDTH, pImage), a_img_result(HEIGHT, WIDTH, pImage);


	//get radius for the 1D filter vector. will be used performing calcs on corresponding image and filter value
	int filter_radius = (static_cast<int>(a_filter.extent[0]) - 1) / 2;
	try
	{
		//apply seperable filter along one dimension on the image (width)
		concurrency::parallel_for_each(a_img.extent, [=, &a_filter](concurrency::index<2> idx) restrict(amp)
			{
				//store data of single dimension convolution run through on buffer array
				a_img_buffer[idx] = convolve<1>(idx, filter_radius, a_img, a_filter);

			});
		//apply the filter on the second dimension (height)
		concurrency::parallel_for_each(a_img.extent, [=, &a_filter](concurrency::index<2> idx) restrict(amp)
			{
				//store data in the final image result array
				a_img_result[idx] = convolve<0>(idx, filter_radius, a_img_buffer, a_filter);
			});

		//set the original image data equal to the results after the filter is applied
		//a_img_result.synchronize();

	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}
}

//compute madlebrot and apply seperable filter convolution
template <class T>
void mandelbrot_convolution_sf_tile(T sigma)
{
	//create necessary variables for filter 
	size_t radius = 5;
	size_t filter_size = radius * 2 + 1;
	std::vector<float> v_filter(radius * 2 + 1);

	//get gaussian distributed filter data 
	float GKernel[5][5];
	FilterCreation(GKernel, sigma);

	// initialize filter values
	//must flatten the GKernel from 2D array into 1D vector
	v_filter = flatten<5, 5>(GKernel);

	//reference to the globally declared image data
	uint32_t* pImage = &(image[0][0]);

	//create array object to encapsulate the filter data
	extent<1> ek(v_filter.size());
	array<float, 1> a_filter(ek, v_filter.begin());

	extent<2> ext(HEIGHT, WIDTH);
	//arrays for use of applying the filters.(buffer is for the first pass through of the seperable filter)
	array_view<uint32_t, 2>  a_img(ext, pImage), a_img_buffer(ext, pImage), a_img_result(ext, pImage);

	//get radius for the 1D filter vector. will be used performing calcs on corresponding image and filter value
	int filter_radius = (static_cast<int>(a_filter.extent[0]) - 1) / 2;

	//(splits the image into blocks for the Tile index to move along)
	extent<2> eRow((((ext[0] - 1) / (TS - 2 * radius)) + 1) * TS, ext[1]);
	extent<2> eCol(ext[0], (((ext[1] - 1) / (TS - 2 * radius)) + 1) * TS);

	try
	{
		//since done with a seperable filter tile needs to be TS x 1 for row and 1 x TS for column
		parallel_for_each(extent<2>(eRow).tile<TS, 1>(), [=,&a_filter](tiled_index<TS, 1> t_idx) restrict(amp)
			{
				convolve_tile<TS, 1, 0>(t_idx, a_img, a_filter, filter_radius, a_img_buffer);
			});
		
		parallel_for_each(extent<2>(eCol).tile<1, TS>(), [=,&a_filter](tiled_index<1, TS> t_idx) restrict(amp)
			{
				convolve_tile<1, TS, 1>(t_idx, a_img_buffer, a_filter, filter_radius, a_img_result);
			});
		
		
	}
	catch (const Concurrency::runtime_exception& ex)
	{
		MessageBoxA(NULL, ex.what(), "Error", MB_ICONERROR);
	}
	
}

//function that templates the dimension index to apply the convoltion filter on
template<int d_idx>
uint32_t convolve(index<2> idx, int radius, const array_view<uint32_t, 2> & img, const array<float, 1> & filter) restrict(amp)
{
	float sum = 0.0f;
	for (int k = -radius; k <= radius; k++)
	{
		//the use of clamp prevents the image/array from going out of bounds
		//some data (first and last thread) will read same data again but that is ok
		int dim = concurrency::direct3d::clamp((int)(idx[d_idx]) + k, 0, (int)(img.extent[d_idx] - 1));

		//get the index to convolve on (0/1 -> width/height)
		index<2> aidx(idx);
		aidx[d_idx] = dim;

		//apply filter
		index<1> kidx(k + radius);
		sum += img[aidx] * filter[kidx];
	}
	//return the image data after going through the filter
	return sum;
}

//function that templates the dimension index to apply the convoltion filter on
template<int d_y, int d_x, int d_idx> //d_y and d_x for getting the index of the tile (for seperable filter use)
void convolve_tile(tiled_index<d_y, d_x> &tidx, const array_view<uint32_t, 2> & img, const array<float, 1> & filter, int radius, array_view<uint32_t, 2> result) restrict(amp)
{
	//allocate local memory for tile to access
	tile_static float local[TS];

	index<2> tile_idx = tidx.tile; //get index of tile
	index<2> local_idx = tidx.local; //get index of image data

	//(TS - 2 * radius) gets rid of parts of the tile that aren't used on the borders of image
	int idx_convolve = (tile_idx[d_idx]) * (TS - 2 * radius) + (int)(local_idx[d_idx]) - radius; //selects pixel to operate on based on tile and local index.
	int max_idx_convolve = img.extent[d_idx]; //only convoloves on the given dimension 
	float sum = 0.0f;

	index<2> a_idx(tile_idx); //holds tile index
	a_idx[d_idx] = concurrency::direct3d::clamp(idx_convolve, 0, max_idx_convolve - 1); //make sure not to go out of bounds
	if (idx_convolve < (max_idx_convolve + radius))
	{
		local[local_idx[d_idx]] = img[a_idx];
	}
	//assures that all data is in tile_static mem before applying filter
	tidx.barrier.wait();

	if ((int)(local_idx[d_idx]) >= radius && (int)(local_idx[d_idx]) < (TS - radius) && idx_convolve < max_idx_convolve)
	{
		for (int k = -radius; k <= radius; k++)
		{
			index<1> k_idx(k + radius);
			sum += local[local_idx[d_idx] + k] * filter[k_idx]; //perform convolution
		}
		result[a_idx] = sum;
	}
}

//test convolution a set amount of iterations and return the median
void testConvolve(float coeffDist, int iter)
{
	cout << "\nCalculating Seperable Filter Convolution... " << endl;
	//create file to put data into for analysis
	ofstream my_file("sf_1024x768.csv");

	//will calculate the median n(iteration #) times
	for(int i = 0; i < iter; i++)
	{
		//vector that will hold all times and will get the median.
		std::vector<float> times;

		for (int j = 0; j < iter; j++)
		{
			//render mandelbrot for the convolution to use
			compute_mandelbrot_tile(-2.0, 1.0, 1.125, -1.125);

			// Start timing
			the_clock::time_point start = the_clock::now();

			//take additional parameter for distribution size
			mandelbrot_convolution_sf(coeffDist);

			// Stop timing
			the_clock::time_point end = the_clock::now();

			// Compute the difference between the two times in milliseconds
			auto time_taken = duration_cast<milliseconds>(end - start).count();
			//cout << "Iteration " << j << " took:  " << time_taken << " ms." << endl;

			//add the time to the vector
			times.push_back((float)time_taken);
		}

		//needed for algorithm to get median
		int median;

		const auto median_it1 = times.begin() + times.size() / 2 - 1;
		const auto median_it2 = times.begin() + times.size() / 2;

		std::nth_element(times.begin(), median_it1, times.end());
		const auto e1 = *median_it1;

		std::nth_element(times.begin(), median_it2, times.end());
		const auto e2 = *median_it2;

		median = (e1 + e2) / 2;

		cout << "Iteration " << i <<" Median time: " << median << "ms" << endl;
		//output median times
		my_file << i << ", " << median << "\n";
	}
}

void testConvolveTile(float coeffDist, int iter)
{
	cout << "\nCalculating Seperable Filter Convolution via Tiles... " << endl;
	ofstream my_file_2("tiled_sf_1024x768_TS_128.csv");

	for (int i = 0; i < iter; i++)
	{
		//vector that will hold all times and will get the median.
		std::vector<float> times;

		for (int j = 0; j < iter; j++)
		{
			//render mandelbrot for the convolution to use
			compute_mandelbrot(-2.0, 1.0, 1.125, -1.125);

			// Start timing
			the_clock::time_point start = the_clock::now();

			//take additional parameter for distribution size
			mandelbrot_convolution_sf_tile(coeffDist);

			// Stop timing
			the_clock::time_point end = the_clock::now();

			// Compute the difference between the two times in milliseconds
			auto time_taken = duration_cast<milliseconds>(end - start).count();
			//cout << "Iteration " << j << " took:  " << time_taken << " ms." << endl;

			//add the time to the vector
			times.push_back((float)time_taken);
		}

		//needed for algorithm to get median
		int median;

		const auto median_it1 = times.begin() + times.size() / 2 - 1;
		const auto median_it2 = times.begin() + times.size() / 2;

		std::nth_element(times.begin(), median_it1, times.end());
		const auto e1 = *median_it1;

		std::nth_element(times.begin(), median_it2, times.end());
		const auto e2 = *median_it2;

		median = (e1 + e2) / 2;

		cout << "Iteration " << i << " Median time: " << median << "ms" << endl;

		//output median times to file
		my_file_2 << i << ", " << median << "\n";
	}
}

void fullTestSuite(float coeffDist)
{
	//how many times to repeat each computation, will be used to get the median
	int iterations = 10;

	cout << "Please wait... " << endl;
	cout << "Generating Mandelbrot Image of " << WIDTH << "x"<< HEIGHT << " Dimensions..." << endl;


	//do an intial run through of the method to setup AMP so the timed 
	//sections do not get affected with setup time
	compute_mandelbrot(-2.0, 1.0, 1.125, -1.125);

	write_tga("before.tga");

	testConvolve(coeffDist, iterations);
	cout << "Done." << endl;

	write_tga("after_sf.tga");


	testConvolveTile(coeffDist, iterations);
	cout << "Done." << endl;

	write_tga("after_tile.tga");
}

int main(int argc, char* argv[])
{
	//check AMP support
	//query_AMP_support();

	std::string input; // used to take input and to be validated before converting to float
	float num;
	bool check = false;

	while (!check)
	{
		cout << "Choose a distribution size for the mandelbrot gaussian filter convolution (reccommended 0.1 to 10.0)" << endl;
		cout << "Please enter a number: ";
		std::cin >> input;

		//checks to see if the input is anything other than a float
		if (input.find_first_not_of("1234567890.-") != std::string::npos) 
		{
			cout << "Invalid input." << endl;
		}
		else
		{
			num = atof(input.c_str()); //convert string into float
			check = true;
		}
	}

	fullTestSuite(num);
	
	cout << "You can find all output files in app folder." << endl;

	return 0;
}





