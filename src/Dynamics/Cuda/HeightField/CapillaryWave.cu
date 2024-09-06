#include "CapillaryWave.h"

#include "SceneGraph.h"

#include <Mapping/HeightFieldToTriangleSet.h>

namespace dyno
{
	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave()
		: Node()
	{
		auto heights = std::make_shared<HeightField<TDataType>>();
		this->stateHeightField()->setDataPtr(heights);
	}

	template<typename TDataType>
	CapillaryWave<TDataType>::~CapillaryWave()
	{
		mDeviceGrid.clear();
		mDeviceGridNext.clear();
	}

	template <typename Coord3D, typename Coord4D>
	__global__ void CW_UpdateHeightDisp(
		DArray2D<Coord3D> displacement,
		DArray2D<Coord4D> dis)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			displacement(i, j).y = dis(i, j).x;
		}
	}

	template <typename Coord>
	__device__ float C_GetU(Coord gp)
	{
		Real h = max(gp.x, 0.0f);
		Real uh = gp.y;

		Real h4 = h * h * h * h;
		return sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));
	}

	template <typename Coord>
	__device__ Real C_GetV(Coord gp)
	{
		Real h = max(gp.x, 0.0f);
		Real vh = gp.z;

		Real h4 = h * h * h * h;
		return sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));
	}

	template <typename Coord>
	__global__ void CW_MoveSimulatedRegion(
		DArray2D<Coord> grid_next,
		DArray2D<Coord> grid,
		int width,
		int height,
		int dx,
		int dy,
		Real level)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < width && j < height)
		{
			int gx = i + 1;
			int gy = j + 1;

			Coord gp = grid(gx, gy);
			Coord gp_init = Coord(level, 0.0f, 0.0f, gp.w);

			int new_i = i - dx;
			int new_j = j - dy;

			if (new_i < 0 || new_i >= width) gp = gp_init;

			new_i = new_i % width;
			if (new_i < 0) new_i = width + new_i;

			if (new_j < 0 || new_j >= height) gp = gp_init;

			new_j = new_j % height;
			if (new_j < 0) new_j = height + new_j;

			grid(new_i + 1, new_j + 1) = gp;
		}
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::moveDynamicRegion(int nx, int ny)
	{
		auto res = this->varResolution()->getValue();

		auto level = this->varWaterLevel()->getValue();

		int extNx = res + 2;
		int extNy = res + 2;

		cuExecute2D(make_uint2(extNx, extNy),
			CW_MoveSimulatedRegion,
			mDeviceGridNext,
			mDeviceGrid,
			res,
			res,
			nx,
			ny,
			level);

		mOriginX += nx;
		mOriginY += ny;
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::resetStates()
	{
		int res = this->varResolution()->getValue();
		Real length = this->varLength()->getValue();

		Real level = this->varWaterLevel()->getValue();

		mRealGridSize = length / res;

		int extNx = res + 2;
		int extNy = res + 2;

		mDeviceGrid.resize(extNx, extNy);
		mDeviceGridNext.resize(extNx, extNy);
		this->stateHeight()->resize(res, res);

		//init grid with initial values
		cuExecute2D(make_uint2(extNx, extNy),
			InitDynamicRegion,
			mDeviceGrid,
			extNx,
			extNy,
			level);

		//init grid_next with initial values
		cuExecute2D(make_uint2(extNx, extNy),
			InitDynamicRegion,
			mDeviceGridNext,
			extNx,
			extNy,
			level);

		auto topo = this->stateHeightField()->getDataPtr();
		topo->setExtents(res, res);
		topo->setGridSpacing(mRealGridSize);
		topo->setOrigin(Coord3D(-0.5 * mRealGridSize * topo->width(), 0, -0.5 * mRealGridSize * topo->height()));

		auto& disp = topo->getDisplacement();

		uint2 extent;
		extent.x = disp.nx();
		extent.y = disp.ny();

		cuExecute2D(extent,
			CW_InitHeightDisp,
			this->stateHeight()->getData(),
			disp,
			mDeviceGrid,
			level);
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateStates()
	{
		Real dt = this->stateTimeStep()->getValue();

		Real level = this->varWaterLevel()->getValue();

		uint res = this->varResolution()->getValue();

		int extNx = res + 2;
		int extNy = res + 2;

		int nStep = 1;
		float timestep = dt / nStep;

		auto scn = this->getSceneGraph();
		auto GRAVITY = scn->getGravity().norm();

		for (int iter = 0; iter < nStep; iter++)
		{
			cuExecute2D(make_uint2(extNx, extNy),
				CW_ImposeBC,
				mDeviceGridNext,
				mDeviceGrid,
				extNx,
				extNy);

			cuExecute2D(make_uint2(res, res),
				CW_OneWaveStep,
				mDeviceGrid,
				mDeviceGridNext,
				res,
				res,
				GRAVITY,
				timestep);
		}

		cuExecute2D(make_uint2(res, res),
			CW_InitHeights,
			this->stateHeight()->getData(),
			mDeviceGrid,
			res,
			mRealGridSize);

		cuExecute2D(make_uint2(res, res),
			CW_InitHeightGrad,
			this->stateHeight()->getData(),
			res);

		//Update topology
		auto topo = this->stateHeightField()->getDataPtr();

		auto& disp = topo->getDisplacement();

		uint2 extent;
		extent.x = disp.nx();
		extent.y = disp.ny();

		cuExecute2D(extent,
			CW_UpdateHeightDisp,
			disp,
			this->stateHeight()->getData());
	}

	template <typename Coord4D>
	__global__ void InitDynamicRegion(DArray2D<Coord4D> grid, int gridwidth, int gridheight, float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			Coord4D gp;
			gp.x = level;
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = 0.0f;

			grid(x, y) = gp;
//			if ((x - 256) * (x - 256) + (y - 256) * (y - 256) <= 2500)  grid(x, y).x = level;
		}
	}

	template <typename Coord4D>
	__global__ void CW_ImposeBC(
		DArray2D<Coord4D> grid_next, 
		DArray2D<Coord4D> grid, 
		int width, 
		int height)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			if (x == 0)
			{
				Coord4D a = grid(1, y);
				grid_next(x, y) = a;
			}
			else if (x == width - 1)
			{
				Coord4D a = grid(width - 2, y);
				grid_next(x, y) = a;
			}
			else if (y == 0)
			{
				Coord4D a = grid(x, 1);
				grid_next(x, y) = a;
			}
			else if (y == height - 1)
			{
				Coord4D a = grid(x, height - 2);
				grid_next(x, y) = a;
			}
			else
			{
				Coord4D a = grid(x, y);
				grid_next(x, y) = a;
			}
		}
	}

	template <typename Coord>
	__host__ __device__ void CW_FixShore(Coord& l, Coord& c, Coord& r)
	{

		if (r.x < 0.0f || l.x < 0.0f || c.x < 0.0f)
		{
			c.x = c.x + l.x + r.x;
			c.x = max(0.0f, c.x);
			l.x = 0.0f;
			r.x = 0.0f;
		}
		float h = c.x;
		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * c.y / (sqrtf(h4 + max(h4, EPSILON)));
		float u = sqrtf(2.0f) * h * c.z / (sqrtf(h4 + max(h4, EPSILON)));

		c.y = u * h;
		c.z = v * h;
	}

	template <typename Coord>
	__host__ __device__ Coord CW_VerticalPotential(Coord gp, float GRAVITY)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));

		Coord G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * h * h;
		G.w = 0.0f;
		return G;
	}

	template <typename Coord>
	__device__ Coord CW_HorizontalPotential(Coord gp, float GRAVITY)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		Coord F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * h * h;
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}

	template <typename Coord>
	__device__ Coord CW_SlopeForce(Coord c, Coord n, Coord e, Coord s, Coord w, float GRAVITY)
	{
		float h = max(c.x, 0.0f);

		Coord H;
		H.x = 0.0f;
		H.y = -GRAVITY * h * (e.w - w.w);
		H.z = -GRAVITY * h * (s.w - n.w);
		H.w = 0.0f;
		return H;
	}

	template <typename Coord4D>
	__global__ void CW_OneWaveStep(
		DArray2D<Coord4D> grid_next, 
		DArray2D<Coord4D> grid, 
		int width, 
		int height, 
		float GRAVITY, 
		float timestep)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			Coord4D center = grid(gridx, gridy);

			Coord4D north = grid(gridx, gridy - 1);

			Coord4D west = grid(gridx - 1, gridy);

			Coord4D south = grid(gridx, gridy + 1);

			Coord4D east = grid(gridx + 1, gridy);

			CW_FixShore(west, center, east);
			CW_FixShore(north, center, south);

			Coord4D u_south = 0.5f * (south + center) - timestep * (CW_VerticalPotential(south, GRAVITY) - CW_VerticalPotential(center, GRAVITY));
			Coord4D u_north = 0.5f * (north + center) - timestep * (CW_VerticalPotential(center, GRAVITY) - CW_VerticalPotential(north, GRAVITY));
			Coord4D u_west = 0.5f * (west + center) - timestep * (CW_HorizontalPotential(center, GRAVITY) - CW_HorizontalPotential(west, GRAVITY));
			Coord4D u_east = 0.5f * (east + center) - timestep * (CW_HorizontalPotential(east, GRAVITY) - CW_HorizontalPotential(center, GRAVITY));

			Coord4D u_center = center + timestep * CW_SlopeForce(center, north, east, south, west, GRAVITY) - timestep * (CW_HorizontalPotential(u_east, GRAVITY) - CW_HorizontalPotential(u_west, GRAVITY)) - timestep * (CW_VerticalPotential(u_south, GRAVITY) - CW_VerticalPotential(u_north, GRAVITY));
			u_center.x = max(0.0f, u_center.x);

			grid_next(gridx, gridy) = u_center;
		}
	}

	template <typename Coord>
	__global__ void CW_InitHeights(
		DArray2D<Coord> height,
		DArray2D<Coord> grid,
		int patchSize,
		float realSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int gridx = i + 1;
			int gridy = j + 1;

			Coord gp = grid(gridx, gridy);
			height(i, j) = gp;
		}
	}

	template <typename Coord4D>
	__global__ void CW_InitHeightGrad(
		DArray2D<Coord4D> height,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int i_minus_one = (i - 1 + patchSize) % patchSize;
			int i_plus_one = (i + 1) % patchSize;
			int j_minus_one = (j - 1 + patchSize) % patchSize;
			int j_plus_one = (j + 1) % patchSize;

			Coord4D Dx = (height(i_plus_one, j) - height(i_minus_one, j)) / 2;
			Coord4D Dz = (height(i, j_plus_one) - height(i, j_minus_one)) / 2;

			height(i, j).z = Dx.y;
			height(i, j).w = Dz.y;
		}
	}

	template <typename Real, typename Coord3D, typename Coord4D>
	__global__ void CW_InitHeightDisp(
		DArray2D<Coord4D> heights,
		DArray2D<Coord3D> displacement,
		DArray2D<Coord4D> grid,
		Real horizon)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			int gridx = i + 1;
			int gridy = j + 1;

			Coord4D gij = grid(gridx, gridy);

			displacement(i, j).x = 0;
			displacement(i, j).y = gij.x + gij.w;
			displacement(i, j).z = 0;

			heights(i, j) = gij;
		}
	}

	DEFINE_CLASS(CapillaryWave);
}