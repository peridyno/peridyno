#pragma once

#include "Vector.h"
#include "VkDeviceArray2D.h"
#include "VkFFT_Defs.h"

namespace dyno
{
	enum VkFFT_Type
	{
		VkFFT_INVERSE = -1,
		VkFFT_FORWARD = 1
	};

	class VkFFT
	{
	public:
		VkFFT();
		~VkFFT();

		static VkFFT* createInstance(VkDeviceArray2D<dyno::Vec2f>& array2d);

		bool update(VkFFT_Type type);

	private:
		bool createContext();

		bool createPipeline(VkDeviceArray2D<dyno::Vec2f>& array2d);

		VkGPU vkGPU = {};
		VkFFTConfiguration configuration = {};
		VkFFTApplication app = {};
	};
}
