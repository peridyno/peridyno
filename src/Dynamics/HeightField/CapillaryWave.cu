#include "CapillaryWave.h"

#include "Topology/HeightField.h"
#include "CapillaryWaveModule.h"

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
		heights->setExtents(size, size);
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
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			int id = displacement.index(i, j);
			/*
			Vec4f Dij = dis[id];

			Coord v;
			v.x = choppiness * Dij.x;
			v.y = Dij.y;
			v.z = choppiness * Dij.z;
			displacement(i, j) = v;
			*/
			displacement(i, j).x = 0;
			displacement(i, j).y += 0.001;
			displacement(i, j).z = 0;
		}
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateTopology()
	{
		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		auto& shifts = topo->getDisplacement();

		uint2 extent;
		extent.x = shifts.nx();
		extent.y = shifts.ny();

		cuExecute2D(extent,
			O_UpdateTopology,
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
		this->animationPipeline()->update();
	}

	DEFINE_CLASS(CapillaryWave);
}