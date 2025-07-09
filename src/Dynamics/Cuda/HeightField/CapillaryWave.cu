#include "CapillaryWave.h"

#include "SceneGraph.h"

#include "Module/NumericalScheme.h"

#include "Mapping/HeightFieldToTriangleSet.h"
#include "GLSurfaceVisualModule.h"
#include <cuda_runtime.h>
#include "ColorMapping.h"

namespace dyno
{
#define Velocity_Limiter Real(60)

	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave()
		: Node()
	{
		this->varViscosity()->setRange(0, 1);

		auto heights = std::make_shared<HeightField<TDataType>>();
		this->stateHeightField()->setDataPtr(heights);

		auto mapper = std::make_shared<HeightFieldToTriangleSet<DataType3f>>();
		this->stateHeightField()->connect(mapper->inHeightField());
		this->graphicsPipeline()->pushModule(mapper);
		
		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(0, 0.2, 1.0));
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
	}

	template<typename TDataType>
	CapillaryWave<TDataType>::~CapillaryWave()
	{
		mDeviceGrid.clear();
		mDeviceGridNext.clear();
		mDeviceGridOld.clear();
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

		this->stateHeight()->resize(res, res);
		this->stateHeight()->reset();

		//init grid with initial values
		cuExecute2D(make_uint2(res, res),
			InitDynamicRegion,
			this->stateHeight()->getData(),
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
			disp,
			this->stateHeight()->getData(),
			level);
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateStates()
	{
		uint frame = this->stateFrameNumber()->getValue();

		Real dt = this->stateTimeStep()->getValue();

		Real level = this->varWaterLevel()->getValue();
		
		auto grid = this->stateHeight()->getData();

		uint res = grid.nx();

		int extNx = res + 2;
		int extNy = res + 2;

		int nStep = 1;
		float timestep = dt / nStep;

		auto scn = this->getSceneGraph();
		auto GRAVITY = scn->getGravity().norm();

		if (mDeviceGrid.nx() != extNx || mDeviceGrid.ny() != extNy)
		{
			mDeviceGrid.resize(extNx, extNy);
			mDeviceGridNext.resize(extNx, extNy);
			mDeviceGridOld.resize(extNy, extNy);
		}

		//init grid_next with initial values
		cuExecute2D(make_uint2(extNx, extNy),
			AssignComputeGrid,
			mDeviceGrid,
			this->stateHeight()->constData());

		mDeviceGridNext.assign(mDeviceGrid);

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

		//A simple viscosity model that does not conserve the total momentum, develop a momentum conserving model in the future.
		mDeviceGridOld.assign(mDeviceGrid);
		for (int iter = 0; iter < 5; iter++)
		{
			cuExecute2D(make_uint2(extNx, extNy),
				CW_ImposeBC,
				mDeviceGridNext,
				mDeviceGrid,
				extNx,
				extNy);

			cuExecute2D(make_uint2(res, res),
				CW_ApplyViscosity,
				mDeviceGrid,
				mDeviceGridNext,
				mDeviceGridOld,
				res,
				res,
				this->varViscosity()->getValue(),
				timestep);
		}

		cuExecute2D(make_uint2(res, res),
			CW_InitHeights,
			this->stateHeight()->getData(),
			mDeviceGrid);

		//Update topology
		auto topo = this->stateHeightField()->getDataPtr();

		auto& disp = topo->getDisplacement();

		uint2 extent;
		extent.x = disp.nx();
		extent.y = disp.ny();

		cuExecute2D(extent,
			CW_InitHeightDisp,
			disp,
			this->stateHeight()->getData(),
			level);

		Node::updateStates();
	}

	template <typename Coord4D>
	__global__ void InitDynamicRegion(
		DArray2D<Coord4D> grid, 
		Real level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < grid.nx() && y < grid.ny())
		{
			Coord4D gp;
			gp.x = level;
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = 0.0f;

			grid(x, y) = gp;
		}
	}

	template <typename Coord4D>
	__global__ void AssignComputeGrid(
		DArray2D<Coord4D> computeGrid, 
		DArray2D<Coord4D> stateGrid)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < stateGrid.nx() && y < stateGrid.ny())
		{
			computeGrid(x + 1, y + 1) = stateGrid(x, y);
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
	__host__ __device__ void CW_FixHorizontalShore(Coord& l, Coord& c, Coord& r)
	{
		if (r.x <= 0)
		{
			r.x = 0;
			r.y = 0;
			r.z = 0;// r.w > c.x + c.w ? -c.z : 0;
			r.w = r.w > c.x + c.w ? c.x + c.w : r.w;
		}

		if (l.x <= 0)
		{
			l.x = 0;
			l.y = 0;
			l.z = 0;//r.w > c.x + c.w ? -c.z : 0;
			l.w = l.w > c.x + c.w ? c.x + c.w : l.w;
		}
	}

	template <typename Coord>
	__host__ __device__ void CW_FixVerticalShore(Coord& l, Coord& c, Coord& r)
	{
		if (r.x <= 0)
		{
			r.x = 0;
			r.y = 0;//r.w > c.x + c.w ? -c.y : 0;
			r.z = 0;
			r.w = r.w > c.x + c.w ? c.x + c.w : r.w;
		}

		if (l.x <= 0)
		{
			l.x = 0;
			l.y = 0;//r.w > c.x + c.w ? -c.y : 0;
			l.z = 0;
			l.w = l.w > c.x + c.w ? c.x + c.w : l.w;
		}
	}

	template <typename Coord>
	__device__ Coord CW_HorizontalPotential(Coord gp, float H, float GRAVITY)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		Coord F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * H * (gp.x + gp.w);
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}

	template <typename Coord>
	__host__ __device__ Coord CW_VerticalPotential(Coord gp, float H, float GRAVITY)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));

		Coord G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * H * (gp.x + gp.w);
		G.w = 0.0f;
		return G;
	}

	template <typename Coord>
	__device__ Coord CW_HorizontalSlope(Coord gp, float H, float GRAVITY)
	{
		Coord F;
		F.x = 0.0f;// u* h;
		F.y = GRAVITY * H * (gp.x + gp.w);
		F.z = 0.0f;
		F.w = 0.0f;
		return F;
	}

	template <typename Coord>
	__host__ __device__ Coord CW_VerticalSlope(Coord gp, float H, float GRAVITY)
	{
		Coord G;
		G.x = 0.0f;
		G.y = 0.0f;
		G.z = GRAVITY * H * (gp.x + gp.w);
		G.w = 0.0f;
		return G;
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

			Real H = max(center.x, 0.0f);

 			//CW_FixHorizontalShore(west, center, east);
 			//CW_FixVerticalShore(north, center, south);

 			//Coord4D u_south = 0.5f * (south + center) - timestep * (CW_VerticalPotential(south, H, GRAVITY) - CW_VerticalPotential(center, H, GRAVITY));
 			//Coord4D u_north = 0.5f * (north + center) - timestep * (CW_VerticalPotential(center, H, GRAVITY) - CW_VerticalPotential(north, H, GRAVITY));
 			//Coord4D u_west = 0.5f * (west + center) - timestep * (CW_HorizontalPotential(center, H, GRAVITY) - CW_HorizontalPotential(west, H, GRAVITY));
 			//Coord4D u_east = 0.5f * (east + center) - timestep * (CW_HorizontalPotential(east, H, GRAVITY) - CW_HorizontalPotential(center, H, GRAVITY));
 
 			//Coord4D u_center = center - timestep * (CW_HorizontalPotential(u_east, H, GRAVITY) - CW_HorizontalPotential(u_west, H, GRAVITY)) - timestep * (CW_VerticalPotential(u_south, H, GRAVITY) - CW_VerticalPotential(u_north, H, GRAVITY));


 			//Coord4D eastflux = CentralUpwindX(center, east, GRAVITY);
 			//Coord4D westflux = CentralUpwindX(west, center, GRAVITY);
 			//Coord4D southflux = CentralUpwindY(center, south, GRAVITY);
 			//Coord4D northflux = CentralUpwindY(north, center, GRAVITY);
 			//Coord4D flux = eastflux - westflux + southflux - northflux;
 
 			//Coord4D u_center = center - timestep * flux - timestep * (CW_HorizontalSlope(east, H, GRAVITY) - CW_HorizontalSlope(west, H, GRAVITY)) - timestep * (CW_VerticalSlope(south, H, GRAVITY) - CW_VerticalSlope(north, H, GRAVITY));


			Coord4D eastflux = FirstOrderUpwindX(center, east, GRAVITY, timestep);
			Coord4D westflux = FirstOrderUpwindX(west, center, GRAVITY, timestep);
			Coord4D southflux = FirstOrderUpwindY(center, south, GRAVITY, timestep);	//Coord4D(0); //
			Coord4D northflux = FirstOrderUpwindY(north, center, GRAVITY, timestep);//Coord4D(0); //
			Coord4D flux = eastflux - westflux + southflux - northflux;

			Coord4D u_center = center - timestep * flux;
			u_center.y += 0.5 * (FirstOrderUpwindPotential(center, east, GRAVITY, timestep) + FirstOrderUpwindPotential(west, center, GRAVITY, timestep));
			u_center.z += 0.5 * (FirstOrderUpwindPotential(center, south, GRAVITY, timestep) + FirstOrderUpwindPotential(north, center, GRAVITY, timestep));

			//Hack: Clamp small value to ensure stability
			Real hc = maximum(u_center.x, 0.0f);
			Real hc4 = hc * hc * hc * hc;
			Real uc = sqrtf(2.0f) * hc * u_center.y / (sqrtf(hc4 + maximum(hc4, Real(0.00001))));
			Real vc = sqrtf(2.0f) * hc * u_center.z / (sqrtf(hc4 + maximum(hc4, Real(0.00001))));

			//Hack: clamp large velocites
			uc = minimum(Velocity_Limiter, maximum(-Velocity_Limiter, uc));
			vc = minimum(Velocity_Limiter, maximum(-Velocity_Limiter, vc));

			grid_next(gridx, gridy) = Coord4D(hc, uc * hc, vc * hc, u_center.w);
		}
	}

	template <typename Real, typename Coord4D>
	__global__ void CW_ApplyViscosity(
		DArray2D<Coord4D> grid_next,
		DArray2D<Coord4D> grid,
		DArray2D<Coord4D> grid_old,
		int width,
		int height,
		Real viscosity,
		Real timestep)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			Coord4D center = grid_old(gridx, gridy);

			Coord4D north = grid(gridx, gridy - 1);

			Coord4D west = grid(gridx - 1, gridy);

			Coord4D south = grid(gridx, gridy + 1);

			Coord4D east = grid(gridx + 1, gridy);

			Real eta = viscosity * timestep;
			Real alpha = 1 / (1 + 4 * eta);

			Real u_center, u_north, u_west, u_south, u_east;
			Real v_center, v_north, v_west, v_south, v_east;

			ComputeVelocity(u_center, v_center, center);
			ComputeVelocity(u_north, v_north, north);
			ComputeVelocity(u_west, v_west, west);
			ComputeVelocity(u_south, v_south, south);
			ComputeVelocity(u_east, v_east, east);

			Real u = alpha * u_center + eta * alpha * (u_north + u_west + u_south + u_east);
			Real v = alpha * v_center + eta * alpha * (v_north + v_west + v_south + v_east);

			grid_next(gridx, gridy) = Coord4D(center.x, u * center.x, v * center.x, center.w);
		}
	}

	template <typename Coord>
	__global__ void CW_InitHeights(
		DArray2D<Coord> height,
		DArray2D<Coord> grid)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < height.nx() && j < height.ny())
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
		DArray2D<Coord3D> displacement,
		DArray2D<Coord4D> grid,
		Real level)
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

	template <typename Real, typename Coord3D>
	__global__ void CW_InitHeightDispLand(
		DArray2D<Coord3D> displacement,
		DArray2D<Coord3D> terrain,
		Real horizon)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			//Real water = max(0.0f, horizon - terrain(i, j).y);

			displacement(i, j).x = 0;
			displacement(i, j).y = horizon;
			displacement(i, j).z = 0;
		}
	}

	DEFINE_CLASS(CapillaryWave);
}