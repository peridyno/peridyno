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
	void Ocean<TDataType>::initialize()
	{
		initWholeRegion();
	}


	template<typename TDataType>
	void Ocean<TDataType>::resetStates()
	{
		m_patch->initialize();
		this->initialize();
	}



	template<typename TDataType>
	void Ocean<TDataType>::initWholeRegion()
	{
		Vec4f* wave = new Vec4f[m_oceanWidth*m_oceanHeight];

		for (int j = 0; j < m_oceanHeight; j++)
		{
			for (int i = 0; i < m_oceanWidth; i++)
			{
				Vec4f vij;
				vij.x = (float)i*m_virtualGridSize;
				vij.y = (float)0;
				vij.z = (float)j*m_virtualGridSize;
				vij.w = (float)1;
				wave[j*m_oceanWidth + i] = vij;
			}
		}

		delete[] wave;
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

		m_eclipsedTime += dt;

		cudaError_t error;
		// make dimension
		int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		// 	Vec4f* oceanVertex = mapOceanVertex();
		// 	rgb* oceanColor = mapOceanColor();

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

