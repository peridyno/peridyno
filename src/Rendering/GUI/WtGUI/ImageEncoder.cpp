#include "ImageEncoder.h"
#include <nvjpeg.h>


template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n", file, line,
			static_cast<unsigned int>(result), func);
		exit(-1);
	}
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

ImageEncoderNV::ImageEncoderNV()
{
	checkCudaErrors(nvjpegCreateSimple(&nvjpegHandle));
	checkCudaErrors(nvjpegEncoderStateCreate(nvjpegHandle, &encoderState, NULL));
	checkCudaErrors(nvjpegEncoderParamsCreate(nvjpegHandle, &encodeParams, NULL));
	// default parameters
	checkCudaErrors(nvjpegEncoderParamsSetQuality(encodeParams, 70, NULL));
	//checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encodeParams, 0, NULL));
	checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encodeParams, NVJPEG_CSS_420, NULL));
	
	// initialize buffer
	cudaBuffer.size = 1920 * 1080 * 3;
	checkCudaErrors(cudaMalloc(&cudaBuffer.ptr, cudaBuffer.size));
}

ImageEncoderNV::~ImageEncoderNV()
{
	checkCudaErrors(nvjpegEncoderParamsDestroy(encodeParams));
	checkCudaErrors(nvjpegEncoderStateDestroy(encoderState));
	checkCudaErrors(nvjpegDestroy(nvjpegHandle));

	checkCudaErrors(cudaFree(cudaBuffer.ptr));
}

void ImageEncoderNV::SetQuality(int quality)
{
	checkCudaErrors(nvjpegEncoderParamsSetQuality(encodeParams, quality, NULL));
}


unsigned long ImageEncoderNV::Encode(const unsigned char* data,
	int width, int height, int pitch,
	std::vector<unsigned char>& buffer)
{
	// reallocate image buffer
	size_t bytes = width * height * 3;
	if (bytes > cudaBuffer.size) {
		cudaBuffer.size = bytes;
		checkCudaErrors(cudaFree(cudaBuffer.ptr)); 
		checkCudaErrors(cudaMalloc(&cudaBuffer.ptr, cudaBuffer.size));
	};

	checkCudaErrors(cudaMemcpy(cudaBuffer.ptr, data, bytes, cudaMemcpyHostToDevice));

	nvjpegImage_t nv_image;
	nv_image.channel[0] = (unsigned char*)cudaBuffer.ptr;
	nv_image.pitch[0] = width * 3;

	checkCudaErrors(nvjpegEncodeImage(nvjpegHandle, encoderState, encodeParams,
		&nv_image, NVJPEG_INPUT_RGBI, width, height, NULL));

	size_t length;
	checkCudaErrors(nvjpegEncodeRetrieveBitstream(nvjpegHandle, encoderState, NULL, &length, NULL));

	// get stream itself
	//checkCudaErrors(cudaStreamSynchronize(stream));

	if (buffer.size() < length) buffer.resize(length);

	checkCudaErrors(nvjpegEncodeRetrieveBitstream(nvjpegHandle, encoderState, buffer.data(), &length, NULL));

	// write stream to file
	//checkCudaErrors(cudaStreamSynchronize(stream));

	return length;
}
