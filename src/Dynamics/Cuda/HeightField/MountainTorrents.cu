#include "MountainTorrents.h"

namespace dyno
{
	template<typename TDataType>
	MountainTorrents<TDataType>::MountainTorrents()
		: CapillaryWave<TDataType>()
	{
		this->stateInitialHeights()->allocate();
	}

	template<typename TDataType>
	MountainTorrents<TDataType>::~MountainTorrents()
	{
	}

	template <typename Coord4D, typename Coord3D>
	__global__ void InitDynamicRegionLand(
		DArray2D<Coord4D> grid,
		DArray2D<Coord3D> displacement,
		int gridwidth,
		int gridheight,
		float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			Coord4D gp;
			gp.x = level > displacement(x, y).y ? level - displacement(x, y).y : 0.0;
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = displacement(x, y).y;

			grid(x, y) = gp;

			//if ((x - 512) * (x - 512) + (y - 440) * (y - 440) <= 1500)  grid(x, y).x = level + 10;
			//if ((x - 128) * (x - 128) + (y - 128) * (y - 128) <= 900)  grid(x, y).x = level + 10;
			//if ((x - 500) * (x - 500) + (y - 500) * (y - 500) <= 2500)  grid(x, y).x = 50-gp.w;
			//if (x <= 256 && (y >= 88 && y < 168))  grid(x, y).x = level + 20;
		}
	}

	template <typename Real, typename Coord3D, typename Coord4D>
	__global__ void InitDynamicRegionForWater(
		DArray2D<Coord4D> grid,
		DArray2D<Coord3D> displacement,
		DArray2D<Real> initialHeights,
		int gridwidth,
		int gridheight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			Coord4D gp;
			gp.x = initialHeights(x, y);
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = displacement(x, y).y;

			grid(x, y) = gp;
		}
	}

	template <typename Coord3D, typename Coord4D>
	__global__ void MT_InitHeightDisp(
		DArray2D<Coord3D> displacement,
		DArray2D<Coord4D> grid,
		float level)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			int gridx = i;
			int gridy = j;

			Coord4D gij = grid(gridx, gridy);

			displacement(i, j).x = 0;
			displacement(i, j).y = gij.x + gij.w;
			displacement(i, j).z = 0;

			if (gij.x <= 0.0f) displacement(i, j).y = level;

		}
	}

	template<typename TDataType>
	void MountainTorrents<TDataType>::resetStates()
	{
		auto terrain = this->getTerrain();

		auto heightOfTerrain = terrain->stateHeightField()->constDataPtr();
		auto& tedisp = heightOfTerrain->getDisplacement();

		auto& initialHeights = this->stateInitialHeights()->constData();

		int nx = tedisp.nx();
		int ny = tedisp.ny();

		int nx_w = initialHeights.nx();
		int ny_w = initialHeights.ny();

		Real level = this->varWaterLevel()->getValue();

		int extNx = nx + 2;
		int extNy = ny + 2;

		this->stateHeight()->resize(nx, ny);
		this->stateHeight()->reset();

		if (nx == nx_w && ny == ny_w)
		{
			cuExecute2D(make_uint2(nx, ny),
				InitDynamicRegionForWater,
				this->stateHeight()->getData(),
				tedisp,
				initialHeights,
				nx,
				ny);
		}
		else
		{
			//init grid with initial values
			cuExecute2D(make_uint2(nx, ny),
				InitDynamicRegionLand,
				this->stateHeight()->getData(),
				tedisp,
				nx,
				ny,
				level);
		}
		

		auto topo = this->stateHeightField()->getDataPtr();
		topo->setExtents(nx, ny);
		topo->setGridSpacing(heightOfTerrain->getGridSpacing());
		topo->setOrigin(heightOfTerrain->getOrigin());
		auto& disp = topo->getDisplacement();

		uint2 extent;
		extent.x = disp.nx();
		extent.y = disp.ny();

		cuExecute2D(extent,
			MT_InitHeightDisp,
			disp,
			this->stateHeight()->getData(),
			level);
	}

	template<typename TDataType>
	void MountainTorrents<TDataType>::updateStates()
	{
		CapillaryWave<TDataType>::updateStates();
	}

	DEFINE_CLASS(MountainTorrents);
}