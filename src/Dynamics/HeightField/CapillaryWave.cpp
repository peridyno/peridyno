#include "CapillaryWave.h"

#include "Topology/HeightField.h"
#include "CapillaryWaveModule.h"

#define cuExecute2DD(size, Func, ...){						\
		uint3 pDims;\
		pDims.x = 8;\
		pDims.y = 8;\
		pDims.z = 8;				\
		dim3 threadsPerBlock(8, 8, 1);		\
		Func << <pDims, threadsPerBlock >> > (				\
		__VA_ARGS__);										\
		cuSynchronize();									\
	}
namespace dyno
{


	//IMPLEMENT_CLASS_1(CapillaryWave, TDataType)

	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave(int size, std::string name)
		: Node()
	{
		auto capillaryWaveModule = std::make_shared<CapillaryWaveModule<TDataType>>();
		this->statePosition()->connect(capillaryWaveModule->statePosition());
		this->animationPipeline()->pushModule(capillaryWaveModule);
	
		auto heights = std::make_shared<HeightField<TDataType>>();
		this->currentTopology()->setDataPtr(heights);

		mResolution = size;

		mChoppiness = 1.0f;
	}

	template<typename TDataType>
	CapillaryWave<TDataType>::~CapillaryWave()
	{
		cudaFree(m_displacement);
	}

	template <typename Coord>
	__global__ void O_UpdateTopology(
		DArray2D<Coord> displacement,
		Vec4f* dis,
		float choppiness)
	{
		printf("ok-\n");
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateTopology()
	{
		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		auto& shifts = topo->getDisplacement();

		uint2 extent;
		extent.x = shifts.nx();
		extent.y = shifts.ny();

		cuExecute2DD(extent,
			O_UpdateTopology
				shifts,
				m_displacement,
				mChoppiness);
	}


	template<typename TDataType>
	void CapillaryWave<TDataType>::resetStates()
	{
		int outputSize = mResolution * mResolution * sizeof(Vec2f);
		cudaMalloc((void**)&m_displacement, mResolution * mResolution * sizeof(Vec4f));
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateStates()
	{
	}

	DEFINE_CLASS(CapillaryWave);
}