#include <iostream>
#include "Ocean.h"
#include <cufft.h>

#include "Topology/HeightField.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

namespace dyno
{

	template<typename TDataType>
	Ocean<TDataType>::Ocean()
		: Node()
	{
		auto heights = std::make_shared<HeightField<TDataType>>();
		this->stateTopology()->setDataPtr(heights);

		m_eclipsedTime = 0;

		m_virtualGridSize = 0.1f;

		m_oceanWidth = m_fft_size * Nx;
		m_oceanHeight = m_fft_size * Ny;

		heights->setExtents(m_oceanHeight, m_oceanHeight);

		m_choppiness = 1.0f;
	}

	template<typename TDataType>
	Ocean<TDataType>::~Ocean()
	{
		
	}

	template<typename TDataType>
	void Ocean<TDataType>::resetStates()
	{
		
		auto m_patch = this->getOceanPatch();
		if (m_patch != nullptr){
			auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());

			auto patch = TypeInfo::cast<HeightField<TDataType>>(m_patch->stateTopology()->getDataPtr());
	
			float h = patch->getGridSpacing();
			topo->setExtents(Nx * patch->width(), Ny * patch->height());
			topo->setGridSpacing(h);
			topo->setOrigin(Vec3f(-0.5*h*topo->width(), 0, -0.5*h*topo->height()));
		
		}
	}

	__global__ void InitOceanWave(
		DArray2D<Vec3f> oceanVertex,
		DArray2D<Vec3f> displacement)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		int width = displacement.nx();
		int height = displacement.ny();

		if (i < width && j < height)
		{
			Vec3f D_ij = displacement(i, j);

			int tiledX = oceanVertex.nx() / displacement.nx();
			int tiledY = oceanVertex.ny() / displacement.ny();
			for (int t = 0; t < tiledX; t++)
			{
				for (int s = 0; s < tiledY; s++)
				{
					int nx = i + t * width;
					int ny = j + s * height;

					oceanVertex(nx, ny) = D_ij;// [ny * oceanWidth + nx] = v;
				}
			}
		}
	}

	__global__ void AddOceanTrails(
		DArray2D<Vec3f> oceanVertex,
		DArray2D<Vec3f> CapillaryWave)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		int width = CapillaryWave.nx();
		int height = CapillaryWave.ny();

		if (i < width && j < height)
		{
			Vec3f C_ij = CapillaryWave(i, j);

			int tiledX = oceanVertex.nx() / CapillaryWave.nx();
			int tiledY = oceanVertex.ny() / CapillaryWave.ny();
			for (int t = 0; t < tiledX; t++)
			{
				for (int s = 0; s < tiledY; s++)
				{
					int nx = i + t * width;
					int ny = j + s * height;

					oceanVertex(nx, ny) += C_ij;
				}
			}
		}
	}

	template<typename TDataType>
	void Ocean<TDataType>::animate(float dt)
	{
		auto m_patch = this->getOceanPatch();

		m_patch->animate(m_eclipsedTime);

		m_eclipsedTime += dt;

		cudaError_t error;
		// make dimension
		int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());

		auto topoPatch = TypeInfo::cast<HeightField<TDataType>>(m_patch->stateTopology()->getDataPtr());
		topo->setGridSpacing(topoPatch->getGridSpacing());
	
		DArray2D<Vec3f> displacement = topoPatch->getDisplacement();
		cuExecute2D(make_uint2(displacement.nx(), displacement.ny()),
			InitOceanWave,
			topo->getDisplacement(),
			displacement);

		auto capillaryWaves = this->getCapillaryWaves();
		for(int i = 0; i < capillaryWaves.size(); i++){
			auto topoCapillaryWave = TypeInfo::cast<HeightField<TDataType>>(capillaryWaves[i]->stateTopology()->getDataPtr());
			
			cuExecute2D(make_uint2(topoCapillaryWave->getDisplacement().nx(), topoCapillaryWave->getDisplacement().ny()),
				AddOceanTrails,
				topo->getDisplacement(),
				topoCapillaryWave->getDisplacement());
		}
		
	}

	template<typename TDataType>
	void Ocean<TDataType>::updateStates()
	{

		this->animate(0.016f);
	}

	template<typename TDataType>
	float Ocean<TDataType>::getPatchLength()
	{
		return m_patchSize;
	}

	template<typename TDataType>
	float Ocean<TDataType>::getGridLength()
	{
		return m_patchSize / m_fft_size;
	}


	DEFINE_CLASS(Ocean);
}

