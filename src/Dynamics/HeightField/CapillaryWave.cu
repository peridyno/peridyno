#include "CapillaryWave.h"

#include "Topology/HeightField.h"

namespace dyno
{
#define GRAVITY 9.83219 * 0.5
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16


	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave(int size, float patchLength, std::string name)
		: Node(name)
	{
		auto heights = std::make_shared<HeightField<TDataType>>();
		heights->setExtents(size, size);
		this->currentTopology()->setDataPtr(heights);

		mResolution = size;
		mChoppiness = 1.0f;

		patchLength = patchLength;
		realGridSize = patchLength / size;

		simulatedRegionWidth = size;
		simulatedRegionHeight = size;

		initialize();
	}
	
	template<typename TDataType>
	CapillaryWave<TDataType>::CapillaryWave(std::string name)
		: Node(name)
	{
	}

	template<typename TDataType>
	CapillaryWave<TDataType>::~CapillaryWave()
	{
		mDisplacement.clear();
		mDeviceGrid.clear();
		mDeviceGridNext.clear();
		mHeight.clear();
		mSource.clear();
		//cudaFree(mWeight);
	}

	template <typename Coord>
	__global__ void O_UpdateTopology(
		DArray2D<Coord> displacement,
		DArray2D<Vec4f> dis,
		float choppiness)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			int id = displacement.index(i, j);
			displacement(i, j).y = dis[id].x;
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
			mHeight,
			mChoppiness);	
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::resetStates()
	{
		mDisplacement.resize(mResolution, mResolution);
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::updateStates()
	{
		compute();
		this->animationPipeline()->update();
	}

	template <typename Coord>
	__global__ void C_InitDynamicRegion(DArray2D<Coord> grid, int gridwidth, int gridheight, int pitch, float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			Coord gp;
			gp.x = level;
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = 0.0f;

			
			grid[y * pitch + x] = gp;
			//grid2Dwrite(grid, x, y, pitch, gp);
			if ((x - 256) * (x - 256) + (y - 256) * (y - 256) <= 2500)  grid[y * pitch + x].x = 10.0f;
		}
	}

	__global__ void C_InitSource(
		DArray2D<Vec2f> source,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			if (i < patchSize / 2 + 3 && i > patchSize / 2 - 3 && j < patchSize / 2 + 3 && j > patchSize / 2 - 3)
			{
				Vec2f uv(1.0f, 1.0f);
				source[i + j * patchSize] = uv;
			}
		}
	}

	template <typename Coord>
	__global__ void C_ImposeBC(DArray2D<Coord> grid_next, DArray2D<Coord> grid, int width, int height, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			if (x == 0)
			{
				Coord a = grid[(y)*pitch + 1];
				grid_next[(y)*pitch + x] = a;
			}
			else if (x == width - 1)
			{
				Coord a = grid[(y)*pitch + width - 2];
				grid_next[(y)*pitch + x] = a;
			}
			else if (y == 0)
			{
				Coord a = grid[(1) * pitch + x];
				grid_next[(y)*pitch + x] = a;
			}
			else if (y == height - 1)
			{
				Coord a = grid[(height - 2) * pitch + x];
				grid_next[(y)*pitch + x] = a;
			}
			else
			{
				Coord a = grid[(y)*pitch + x];
				grid_next[(y)*pitch + x] = a;
			}
		}
	}

	template <typename Coord>
	__host__ __device__ void C_FixShore(Coord& l, Coord& c, Coord& r)
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
	__host__ __device__ Coord C_VerticalPotential(Coord gp)
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
	__device__ Coord C_HorizontalPotential(Coord gp)
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
	__device__ Coord C_SlopeForce(Coord c, Coord n, Coord e, Coord s, Coord w)
	{
		float h = max(c.x, 0.0f);

		Coord H;
		H.x = 0.0f;
		H.y = -GRAVITY * h * (e.w - w.w);
		H.z = -GRAVITY * h * (s.w - n.w);
		H.w = 0.0f;
		return H;
	}

	template <typename Coord>
	__global__ void C_OneWaveStep(DArray2D<Coord> grid_next, DArray2D<Coord> grid, int width, int height, float timestep, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			Coord center = grid[gridx + pitch * gridy];

			Coord north = grid[gridx + pitch * (gridy - 1)];

			Coord west = grid[gridx - 1 + pitch * gridy];

			Coord south = grid[gridx + pitch * (gridy + 1)];

			Coord east = grid[gridx + 1 + pitch * gridy];

			C_FixShore(west, center, east);
			C_FixShore(north, center, south);

			Coord u_south = 0.5f * (south + center) - timestep * (C_VerticalPotential(south) - C_VerticalPotential(center));
			Coord u_north = 0.5f * (north + center) - timestep * (C_VerticalPotential(center) - C_VerticalPotential(north));
			Coord u_west = 0.5f * (west + center) - timestep * (C_HorizontalPotential(center) - C_HorizontalPotential(west));
			Coord u_east = 0.5f * (east + center) - timestep * (C_HorizontalPotential(east) - C_HorizontalPotential(center));

			Coord u_center = center + timestep * C_SlopeForce(center, north, east, south, west) - timestep * (C_HorizontalPotential(u_east) - C_HorizontalPotential(u_west)) - timestep * (C_VerticalPotential(u_south) - C_VerticalPotential(u_north));
			u_center.x = max(0.0f, u_center.x);

			grid_next[gridx + gridy * pitch] = u_center;
		}
	}

	template <typename Coord>
	__global__ void C_InitHeightField(
		DArray2D<Coord> height,
		DArray2D<Coord> grid,
		int patchSize,
		float horizon,
		float realSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int gridx = i + 1;
			int gridy = j + 1;

			Coord gp = grid[gridx + patchSize * gridy];
			height[i + j * patchSize].x = gp.x - horizon;

			float d = sqrtf((i - patchSize / 2) * (i - patchSize / 2) + (j - patchSize / 2) * (j - patchSize / 2));
			float q = d / (0.49f * patchSize);

			float weight = q < 1.0f ? 1.0f - q * q : 0.0f;
			height[i + j * patchSize].y = 1.3f * realSize * sinf(3.0f * weight * height[i + j * patchSize].x * 0.5f * M_PI);

			// x component stores the original height, y component stores the normalized height, z component stores the X gradient, w component stores the Z gradient;
		}
	}

	template <typename Coord>
	__global__ void C_InitHeightGrad(
		DArray2D<Coord> height,
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

			Coord Dx = (height[i_plus_one + j * patchSize] - height[i_minus_one + j * patchSize]) / 2;
			Coord Dz = (height[i + j_plus_one * patchSize] - height[i + j_minus_one * patchSize]) / 2;

			height[i + patchSize * j].z = Dx.y;
			height[i + patchSize * j].w = Dz.y;
		}
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::compute()
	{
		float dt = 0.016f;

		int extNx = simulatedRegionWidth + 2;
		int extNy = simulatedRegionHeight + 2;

		cudaError_t error;
		// make dimension
		int x = (simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		int x1 = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y1 = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock1(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid1(x1, y1);

		int nStep = 1;
		float timestep = dt / nStep;

		for (int iter = 0; iter < nStep; iter++)
		{
			C_ImposeBC << < blocksPerGrid1, threadsPerBlock1 >> > (mDeviceGridNext, mDeviceGrid, extNx, extNy, gridPitch);
			swapDeviceGrid();
			//synchronCheck;

			C_OneWaveStep << < blocksPerGrid, threadsPerBlock >> > (
				mDeviceGridNext,
				mDeviceGrid,
				simulatedRegionWidth,
				simulatedRegionHeight,
				1.0f * timestep,
				gridPitch);
			swapDeviceGrid();
			//synchronCheck;
		}

		C_InitHeightField << < blocksPerGrid, threadsPerBlock >> > (mHeight, mDeviceGrid, simulatedRegionWidth, horizon, realGridSize);
		//synchronCheck;

		C_InitHeightGrad << < blocksPerGrid, threadsPerBlock >> > (mHeight, simulatedRegionWidth);
		//synchronCheck;
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::initialize() {
		initDynamicRegion();

		initSource();

		initHeightPosition();
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::initDynamicRegion() {

		int extNx = simulatedRegionWidth + 2;
		int extNy = simulatedRegionHeight + 2;

		mDeviceGrid.resize(extNx, extNy);
		mDeviceGridNext.resize(extNx, extNy);
		mHeight.resize(simulatedRegionWidth, simulatedRegionHeight);

		gridPitch = mResolution;

		int x = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		//init grid with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (mDeviceGrid, extNx, extNy, gridPitch, horizon);
		//synchronCheck;

		//init grid_next with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (mDeviceGridNext, extNx, extNy, gridPitch, horizon);
		//synchronCheck;

	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::initSource() {

		int sizeInBytes = simulatedRegionWidth * simulatedRegionHeight * sizeof(float2);

		mSource.resize(simulatedRegionWidth, simulatedRegionHeight);

		cuSafeCall(cudaMalloc(&mWeight, simulatedRegionWidth * simulatedRegionHeight * sizeof(float)));

		int x = (simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);
		C_InitSource << < blocksPerGrid, threadsPerBlock >> > (mSource, simulatedRegionWidth);
		resetSource();
		//synchronCheck;
	}

	template <typename Coord>
	__global__ void O_InitHeightPosition(
		DArray2D<Coord> displacement,
		float horizon)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			int id = displacement.index(i, j);
			displacement(i, j).x = 0;
			displacement(i, j).y = horizon;
			displacement(i, j).z = 0;

			if ((i - 256) * (i - 256) + (j - 256) * (j - 256) <= 2500)  displacement(i, j).y = 10.0f;
		}
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::initHeightPosition() {
		auto topo = TypeInfo::cast<HeightField<TDataType>>(this->currentTopology()->getDataPtr());

		auto& shifts = topo->getDisplacement();

		uint2 extent;
		extent.x = shifts.nx();
		extent.y = shifts.ny();


		cuExecute2D(extent,
			O_InitHeightPosition,
			shifts,
			horizon);
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::resetSource()
	{
		cudaMemset(mWeight, 0, simulatedRegionWidth * simulatedRegionHeight * sizeof(float));
	}

	template<typename TDataType>
	void CapillaryWave<TDataType>::swapDeviceGrid()
	{
		DArray2D<Vec4f> grid_helper = mDeviceGrid;
		mDeviceGrid = mDeviceGridNext;
		mDeviceGridNext = grid_helper;
	}

	DEFINE_CLASS(CapillaryWave);
}