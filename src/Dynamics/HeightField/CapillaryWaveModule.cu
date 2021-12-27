#include "CapillaryWaveModule.h"
#include "Node.h"
#include "Matrix/MatrixFunc.h"
#include "ParticleSystem/Kernel.h"

#include "cuda_helper_math.h"
//#include <cuda_gl_interop.h>  
//#include <cufft.h>
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define grid2Dwrite(array, x, y, pitch, value) array[(y)*pitch+x] = value
#define grid2Dread(array, x, y, pitch) array[(y)*pitch+x]

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
namespace dyno
{
	IMPLEMENT_CLASS_1(CapillaryWaveModule, TDataType)


	void printDeviceToHost2D(Vec2f* flt, int size) {
		Vec2f* host_h0 = (Vec2f*)malloc(size * sizeof(Vec2f));
		cuSafeCall(cudaMemcpy(host_h0, flt, size * sizeof(Vec2f), cudaMemcpyDeviceToHost));
		printf("begin2D--------\n");
		for (int i = 0; i < size; i++) {
			if ((i / 4) % 100000 == 0)
				printf("id[%d].x = %f, id[%d].y = %f\n", i, host_h0[i].x, i, host_h0[i].y);
		}
		printf("end--------\n");
	}

	void printDeviceToHost4D(Vec4f* flt, int Nx, int Ny) {
		Vec4f* host_h0 = (Vec4f*)malloc(Nx * Ny * sizeof(Vec4f));
		cuSafeCall(cudaMemcpy(host_h0, flt, Nx * Ny * sizeof(Vec4f), cudaMemcpyDeviceToHost));
		printf("begin4D--------\n");
		for (int i = 0; i < Nx * Ny; i += 4) {
			if (i % 100000 == 0) {
				printf("id[%d].x y z w = (%f, %f, %f, %f)\n", i, host_h0[i].x, host_h0[i].y, host_h0[i].z, host_h0[i].w);
				printf("id[%d].x y z w = (%f, %f, %f, %f)\n", i + 1, host_h0[i + 1].x, host_h0[i + 1].y, host_h0[i + 1].z, host_h0[i + 1].w);
				printf("id[%d].x y z w = (%f, %f, %f, %f)\n", i + 2, host_h0[i + 2].x, host_h0[i + 2].y, host_h0[i + 2].z, host_h0[i + 2].w);
				printf("id[%d].x y z w = (%f, %f, %f, %f)\n", i + 3, host_h0[i + 3].x, host_h0[i + 3].y, host_h0[i + 3].z, host_h0[i + 3].w);
			}
		}
		printf("end--------\n");
	}

	template<typename TDataType>
	CapillaryWaveModule<TDataType>::~CapillaryWaveModule() {
		printf("CapillaryWaveModule ~construction \n");

		cudaFree(m_device_grid);
		cudaFree(m_device_grid_next);
		cudaFree(m_height);
		cudaFree(m_source);
		cudaFree(m_weight);
	}
	template<typename TDataType>
	CapillaryWaveModule<TDataType>::CapillaryWaveModule() {
		printf("CapillaryWaveModule construction \n");
	}
	template<typename TDataType>
	CapillaryWaveModule<TDataType>::CapillaryWaveModule(int size, float patchLength) {
		printf("CapillaryWaveModule construction1  \n");

		m_patch_length = patchLength;
		m_realGridSize = patchLength / size;

		m_simulatedRegionWidth = size;
		m_simulatedRegionHeight = size;

		initialize();
	}

	__global__ void C_InitDynamicRegion(Vec4f* grid, int gridwidth, int gridheight, int pitch, float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			Vec4f gp;
			gp.x = level;
			//gp.y = 0.0f;
			gp.y = float((x+y* pitch)%10);
			gp.z = 0.0f;
			gp.w = 0.0f;

			grid[y * pitch + x] = gp;
			//grid2Dwrite(grid, x, y, pitch, gp);
			if ((x - 256) * (x - 256) + (y - 256) * (y - 256) <= 2500)  grid[y * pitch + x].x = 2.5;
		}
	}

	__global__ void C_InitSource(
		Vec2f* source,
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

	__global__ void C_ImposeBC(Vec4f* grid_next, Vec4f* grid, int width, int height, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			if (x == 0)
			{
				Vec4f a = grid[(y)*pitch + 1];
				grid_next[(y)*pitch + x] = a;
				
				//Vec4f a = grid2Dread(grid, 1, y, pitch);
				//grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (x == width - 1)
			{
				Vec4f a = grid[(y)*pitch + width - 2];
				grid_next[(y)*pitch + x] = a;

				//Vec4f a = grid2Dread(grid, width - 2, y, pitch);
				//grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (y == 0)
			{
				Vec4f a = grid[(1)*pitch + x];
				grid_next[(y)*pitch + x] = a;

				//Vec4f a = grid2Dread(grid, x, 1, pitch);
				//grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (y == height - 1)
			{
				Vec4f a = grid[(height - 2) * pitch + x];
				grid_next[(y)*pitch + x] = a;

				//Vec4f a = grid2Dread(grid, x, height - 2, pitch);
				//grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else
			{
				Vec4f a = grid[(y) * pitch + x];
				grid_next[(y)*pitch + x] = a;

				//Vec4f a = grid2Dread(grid, x, y, pitch);
				//grid2Dwrite(grid_next, x, y, pitch, a);
			}
		}
	}
	__host__ __device__ void C_FixShore(Vec4f& l, Vec4f& c, Vec4f& r)
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

	__host__ __device__ Vec4f C_VerticalPotential(Vec4f gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));

		Vec4f G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * h * h;
		G.w = 0.0f;
		return G;
	}

	__device__ Vec4f C_HorizontalPotential(Vec4f gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		Vec4f F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * h * h;
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}

	__device__ Vec4f C_SlopeForce(Vec4f c, Vec4f n, Vec4f e, Vec4f s, Vec4f w)
	{
		float h = max(c.x, 0.0f);

		Vec4f H;
		H.x = 0.0f;
		H.y = -GRAVITY * h * (e.w - w.w);
		H.z = -GRAVITY * h * (s.w - n.w);
		H.w = 0.0f;
		return H;
	}

	__global__ void C_OneWaveStep(Vec4f* grid_next, Vec4f* grid, int width, int height, float timestep, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			Vec4f center = grid[gridx+ pitch*gridy];

			Vec4f north = grid[gridx+ pitch * (gridy - 1)];

			Vec4f west = grid[gridx - 1+ pitch * gridy];

			Vec4f south = grid[gridx+ pitch * (gridy + 1)];

			Vec4f east = grid[gridx + 1+ pitch * gridy];

			C_FixShore(west, center, east);
			C_FixShore(north, center, south);

			Vec4f u_south = 0.5f * (south + center) - timestep * (C_VerticalPotential(south) - C_VerticalPotential(center));
			Vec4f u_north = 0.5f * (north + center) - timestep * (C_VerticalPotential(center) - C_VerticalPotential(north));
			Vec4f u_west = 0.5f * (west + center) - timestep * (C_HorizontalPotential(center) - C_HorizontalPotential(west));
			Vec4f u_east = 0.5f * (east + center) - timestep * (C_HorizontalPotential(east) - C_HorizontalPotential(center));

			Vec4f u_center = center + timestep * C_SlopeForce(center, north, east, south, west) - timestep * (C_HorizontalPotential(u_east) - C_HorizontalPotential(u_west)) - timestep * (C_VerticalPotential(u_south) - C_VerticalPotential(u_north));
			u_center.x = max(0.0f, u_center.x);

			grid_next[gridx+ gridy*pitch]= u_center;
		}
	}

	__global__ void C_InitHeightField(
		Vec4f* height,
		Vec4f* grid,
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

			Vec4f gp = grid[gridx + patchSize * gridy];
			height[i + j * patchSize].x = gp.x - horizon;

			float d = sqrtf((i - patchSize / 2) * (i - patchSize / 2) + (j - patchSize / 2) * (j - patchSize / 2));
			float q = d / (0.49f * patchSize);

			float weight = q < 1.0f ? 1.0f - q * q : 0.0f;
			height[i + j * patchSize].y = 1.3f * realSize * sinf(3.0f * weight * height[i + j * patchSize].x * 0.5f * M_PI);

			// x component stores the original height, y component stores the normalized height, z component stores the X gradient, w component stores the Z gradient;
		}
	}
	__global__ void C_InitHeightGrad(
		Vec4f* height,
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

			Vec4f Dx = (height[i_plus_one + j * patchSize] - height[i_minus_one + j * patchSize]) / 2;
			Vec4f Dz = (height[i + j_plus_one * patchSize] - height[i + j_minus_one * patchSize]) / 2;

			height[i + patchSize * j].z = Dx.y;
			height[i + patchSize * j].w = Dz.y;
		}
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::compute()
	{
		printf("compute \n");

		float dt = 0.016f;

		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		cudaError_t error;
		// make dimension
		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		int x1 = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y1 = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock1(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid1(x1, y1);

		int nStep = 1;
		float timestep = dt / nStep;

		//printDeviceToHost4D(m_device_grid_next, extNx, extNy);

		for (int iter = 0; iter < nStep; iter++)
		{
			//cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
			C_ImposeBC << < blocksPerGrid1, threadsPerBlock1 >> > (m_device_grid_next, m_device_grid, extNx, extNy, m_grid_pitch);
			swapDeviceGrid();
			synchronCheck;
			
			//cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
			C_OneWaveStep << < blocksPerGrid, threadsPerBlock >> > (
				m_device_grid_next,
				m_device_grid,
				m_simulatedRegionWidth,
				m_simulatedRegionHeight,
				1.0f * timestep,
				m_grid_pitch);
			swapDeviceGrid();
			synchronCheck;
		}
		
		//printDeviceToHost4D(m_device_grid_next, extNx, extNy);
		C_InitHeightField << < blocksPerGrid, threadsPerBlock >> > (m_height, m_device_grid, m_simulatedRegionWidth, m_horizon, m_realGridSize);
		synchronCheck;

		//printDeviceToHost4D(m_height, m_simulatedRegionWidth, m_simulatedRegionWidth);

		C_InitHeightGrad << < blocksPerGrid, threadsPerBlock >> > (m_height, m_simulatedRegionWidth);
		synchronCheck;

		//printDeviceToHost4D(m_height, m_simulatedRegionWidth, m_simulatedRegionWidth);


	}

	
	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::initialize() {
		printf("CapillaryWaveModule initialize \n");

		initDynamicRegion();

		initSource();
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::initDynamicRegion() {
		printf("CapillaryWaveModule initDynamicRegion \n");

		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		size_t pitch;
		cuSafeCall(cudaMallocPitch(&m_device_grid, &pitch, extNx * sizeof(Vec4f), extNy));
		cuSafeCall(cudaMallocPitch(&m_device_grid_next, &pitch, extNx * sizeof(Vec4f), extNy));

		cuSafeCall(cudaMalloc((void **)&m_height, m_simulatedRegionWidth*m_simulatedRegionWidth*sizeof(Vec4f)));

		m_grid_pitch = pitch / sizeof(Vec4f);

		m_grid_pitch = 514;

		int x = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);
	
		//init grid with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (m_device_grid, extNx, extNy, m_grid_pitch, m_horizon);
		synchronCheck;

		//init grid_next with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (m_device_grid_next, extNx, extNy, m_grid_pitch, m_horizon);
		synchronCheck;
	
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::initSource() {
		printf("CapillaryWaveModule initSource \n");

		int sizeInBytes = m_simulatedRegionWidth * m_simulatedRegionHeight * sizeof(float2);

		cuSafeCall(cudaMalloc(&m_source, sizeInBytes));
		cuSafeCall(cudaMalloc(&m_weight, m_simulatedRegionWidth * m_simulatedRegionHeight * sizeof(float)));
		cuSafeCall(cudaMemset(m_source, 0, sizeInBytes));

		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);
		C_InitSource << < blocksPerGrid, threadsPerBlock >> > (m_source, m_simulatedRegionWidth);
		resetSource();
		synchronCheck;
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::resetSource()
	{
		cudaMemset(m_source, 0, m_simulatedRegionWidth * m_simulatedRegionHeight * sizeof(float2));
		cudaMemset(m_weight, 0, m_simulatedRegionWidth * m_simulatedRegionHeight * sizeof(float));
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::swapDeviceGrid()
	{
		Vec4f* grid_helper = m_device_grid;
		m_device_grid = m_device_grid_next;
		m_device_grid_next = grid_helper;
	}

	DEFINE_CLASS(CapillaryWaveModule);
}