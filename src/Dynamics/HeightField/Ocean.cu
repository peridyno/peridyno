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
	}

	template<typename TDataType>
	Ocean<TDataType>::~Ocean()
	{
		
	}

	__global__ void O_InitOceanWave2(
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

					oceanVertex(nx, ny).y += 10;// [ny * oceanWidth + nx] = v;
					displacement(i, j).y = oceanVertex(nx, ny).y;

				}
			}
		}
	}

	template<typename TDataType>
	void Ocean<TDataType>::resetStates()
	{
		auto m_patch = this->getOceanPatch();

		m_patch->initialize();

		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		auto patch = TypeInfo::cast<HeightField<TDataType>>(m_patch->currentTopology()->getDataPtr());

		float h = patch->getGridSpacing();
		topo->setExtents(Nx * patch->width(), Ny * patch->height());
		topo->setGridSpacing(h);
		topo->setOrigin(Vec3f(-0.5*h*topo->width(), 0, -0.5*h*topo->height()));



		int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		/*
		O_InitOceanWave2 << < blocksPerGrid, threadsPerBlock >> > (
			topo->getDisplacement(),
			patch->getDisplacement());
		printf("99999999999999999\n");*/
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
		auto m_patch = this->getOceanPatch();

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

		addOceanTrails(topo->getDisplacement());
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

	__global__ void O_AddCapillaryWave(
		DArray2D<Vec3f> oceanVertex,
		DArray2D<Vec4f> heightfield,
		int waveGridSize,
		int oceanWidth,
		int oceanHeight,
		int originX,
		int originY,
		float waveSpacing,
		float oceanSpacing,
		float horizon,
		float realSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i < waveGridSize && j < waveGridSize)
		{
			float d = sqrtf((i - waveGridSize / 2) * (i - waveGridSize / 2) + (j - waveGridSize / 2) * (j - waveGridSize / 2));
			float q = d / (0.49f * waveGridSize);

			float weight = q < 1.0f ? 1.0f - q * q : 0.0f;

			int oi = (i + originX) * waveSpacing / oceanSpacing;
			int oj = (j + originY) * waveSpacing / oceanSpacing;

			if (oi > 0 && oi < oceanWidth && oj > 0 && oj < oceanHeight)
			{
				int ocean_id = oj * oceanWidth + oi;
				int hf_id = j * waveGridSize + i;
				float h_ij = heightfield[hf_id].x;
				Vec3f o_ij = oceanVertex[ocean_id];

				float value = sin(3.0f * weight * h_ij * 0.5f * M_PI);
				o_ij.y += realSize * value;// 3.0f*weight*realSize*h_ij;

				oceanVertex[ocean_id] = o_ij;
			}
		}
	}

	template<typename TDataType>
	void Ocean<TDataType>::addOceanTrails(DArray2D<Vec3f> oceanVertex)
	{

		int x = (m_fft_size + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_fft_size + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);
	
		auto trails = this->getCapillaryWaves();
		for (size_t i = 0; i < trails.size(); i++)
		{
			auto trail = trails[i];
	
			O_AddCapillaryWave << < blocksPerGrid, threadsPerBlock >> > (
				oceanVertex,
				trail->getHeightField(),
				trail->getGridSize(),
				m_oceanWidth,
				m_oceanHeight,
				trail->getOriginX(),
				trail->getOriginZ(),
				trail->getRealGridSize(),
				getGridLength(),
				trail->getHorizon(),
				0.5f);
		}

	}
	DEFINE_CLASS(Ocean);
}

