#pragma once
#include <vector>
#include <cstddef>

class ImageEncoder
{
public:
	virtual void SetQuality(int quality) = 0;
	virtual unsigned long Encode(const unsigned char* data,
		int width, int height, int pitch,
		std::vector<unsigned char>& buffer) = 0;
};

// encode image with nvJPG
struct nvjpegHandle;
struct nvjpegEncoderState;
struct nvjpegEncoderParams;
class ImageEncoderNV : public ImageEncoder
{
public:
	ImageEncoderNV();
	~ImageEncoderNV();

public:
	virtual void SetQuality(int quality) override;
	virtual unsigned long Encode(const unsigned char* data,
		int width, int height, int pitch,
		std::vector<unsigned char>& buffer) override;

private:
	nvjpegHandle* m_nvjpegHandle;
	nvjpegEncoderState* encoderState;
	nvjpegEncoderParams* encodeParams;
	//cudaStream_t stream = NULL;

	// cuda buffer
	struct {
		void* ptr;
		size_t	size;
	} cudaBuffer;
};
