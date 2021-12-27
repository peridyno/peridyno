#include "CapillaryWaveModule.h"
#include "Node.h"
#include "Matrix/MatrixFunc.h"
#include "ParticleSystem/Kernel.h"

#include "cuda_helper_math.h"
//#include <cuda_gl_interop.h>  
//#include <cufft.h>
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define grid2Dwrite(array, x, y, pitch, value) array[(y)*pitch+x] = value
#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
namespace dyno
{
	IMPLEMENT_CLASS_1(CapillaryWaveModule, TDataType)

		template<typename TDataType>
	void CapillaryWaveModule<TDataType>::compute()
	{
		printf("compute \n");
	}
	template<typename TDataType>
	CapillaryWaveModule<TDataType>::CapillaryWaveModule() {
		printf("CapillaryWaveModule construction \n");
	}

	DEFINE_CLASS(CapillaryWaveModule);
}