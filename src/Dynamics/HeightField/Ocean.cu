#include "Ocean.h"

namespace dyno
{
	template<typename TDataType>
	Ocean<TDataType>::Ocean()
		: Node()
	{
		auto heights = std::make_shared<HeightField<TDataType>>();
		this->stateTopology()->setDataPtr(heights);
	}

	template<typename TDataType>
	Ocean<TDataType>::~Ocean()
	{
		this->varExtentX()->setRange(1, 10);
		this->varExtentZ()->setRange(1, 10);
	}

	template<typename TDataType>
	void Ocean<TDataType>::resetStates()
	{
		auto patch = this->getOceanPatch();

		auto Nx = this->varExtentX()->getData();
		auto Nz = this->varExtentZ()->getData();

		auto patchHeights = TypeInfo::cast<HeightField<TDataType>>(patch->stateTopology()->getDataPtr());
		auto oceanHeights = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());

		Real h = patchHeights->getGridSpacing();
		oceanHeights->setExtents(Nx * patchHeights->width(), Nz * patchHeights->height());
		oceanHeights->setGridSpacing(h);
		oceanHeights->setOrigin(Vec3f(-0.5 * h * oceanHeights->width(), 0, -0.5 * h * oceanHeights->height()));
	}

	template<typename Coord>
	__global__ void O_InitOceanWave(
		DArray2D<Coord> oceanVertex,
		DArray2D<Coord> displacement)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		int width = displacement.nx();
		int height = displacement.ny();

		if (i < width && j < height)
		{
			Coord D_ij = displacement(i, j);

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

	template<typename Coord>
	__global__ void O_AddOceanTrails(
		DArray2D<Coord> oceanVertex,
		DArray2D<Coord> CapillaryWave)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		int width = CapillaryWave.nx();
		int height = CapillaryWave.ny();

		if (i < width && j < height)
		{
			Coord C_ij = CapillaryWave(i, j);

			int tiledX = oceanVertex.nx() / CapillaryWave.nx();
			int tiledY = oceanVertex.ny() / CapillaryWave.ny();

			//TODO: correct the position
			int nx = i;
			int ny = j;

			oceanVertex(nx, ny) += C_ij;
		}
	}

	template<typename TDataType>
	void Ocean<TDataType>::updateStates()
	{
		auto patch = this->getOceanPatch();

		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->stateTopology()->getDataPtr());

		auto patchHeights = TypeInfo::cast<HeightField<TDataType>>(patch->stateTopology()->getDataPtr());

		DArray2D<Coord>& patchDisp = patchHeights->getDisplacement();
		cuExecute2D(make_uint2(patchDisp.nx(), patchDisp.ny()),
			O_InitOceanWave,
			topo->getDisplacement(),
			patchDisp);

		auto& waves = this->getCapillaryWaves();
		if (waves.size() > 0)
		{
			for (int i = 0; i < waves.size(); i++) {
				auto wave = TypeInfo::cast<HeightField<TDataType>>(waves[i]->stateTopology()->getDataPtr());

				auto& waveDisp = wave->getDisplacement();

				cuExecute2D(make_uint2(waveDisp.nx(), waveDisp.ny()),
					O_AddOceanTrails,
					topo->getDisplacement(),
					waveDisp);
			}
		}
	}

	template<typename TDataType>
	bool Ocean<TDataType>::validateInputs()
	{
		return this->importOceanPatch() != nullptr;
	}

	DEFINE_CLASS(Ocean);
}

