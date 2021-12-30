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
		this->currentTopology()->setDataPtr(heights);

		m_eclipsedTime = 0.0f;

		m_virtualGridSize = 0.1f;

		m_oceanWidth = m_fft_size * Nx;
		m_oceanHeight = m_fft_size * Ny;

		heights->setExtents(m_oceanHeight, m_oceanHeight);

		m_choppiness = 1.0f;

		m_patch = new OceanPatch<DataType3f>(m_fft_size, m_patchSize, m_windType);

		m_realGridSize = m_patch->getGridLength();
	}

	template<typename TDataType>
	Ocean<TDataType>::~Ocean()
	{
		delete m_patch;
	}

	template<typename TDataType>
	void Ocean<TDataType>::resetStates()
	{
		m_patch->initialize();

		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		auto patch = TypeInfo::cast<HeightField<TDataType>>(m_patch->currentTopology()->getDataPtr());

		float h = patch->getGridSpacing();
		topo->setExtents(Nx * patch->width(), Ny * patch->height());
		topo->setGridSpacing(h);
		topo->setOrigin(Vec3f(-0.5*h*topo->width(), 0, -0.5*h*topo->height()));
	}

	__global__ void O_InitOceanWave(
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

	template<typename TDataType>
	void Ocean<TDataType>::animate(float dt)
	{
		m_patch->animate(m_eclipsedTime);
		m_patch->update();

		m_eclipsedTime += dt;

		cudaError_t error;
		// make dimension
		int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		auto topoPatch = TypeInfo::cast<HeightField<TDataType>>(m_patch->currentTopology()->getDataPtr());

		topo->setGridSpacing(topoPatch->getGridSpacing());

		O_InitOceanWave << < blocksPerGrid, threadsPerBlock >> > (
			topo->getDisplacement(),
			topoPatch->getDisplacement());
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

