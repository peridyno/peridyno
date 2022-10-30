#pragma once
#include <curand_kernel.h> 
#include "Platform.h"

namespace dyno {

#define DIV 10000

	class RandNumber
	{
	public:
		GPU_FUNC RandNumber(int seed)
		{
			curand_init(seed, 0, 0, &s);
		}
		GPU_FUNC ~RandNumber() {};

		/*!
		*	\brief	Generate a float number ranging from 0 to 1.
		*/
		GPU_FUNC float Generate()
		{
			return curand_uniform(&s);
		}

	private:
		curandState s;
	};
}

