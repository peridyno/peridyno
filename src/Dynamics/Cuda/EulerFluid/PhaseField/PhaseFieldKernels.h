#pragma once

#include "Platform.h"
#include "DataTypes.h"
#include "Object.h"
#include "Vector.h"
#include "Array/Array.h"
#include "Array/Array3D.h"

//#define INPUT_BRUSH

namespace dyno{

#define RHO1 1000.0f
#define RHO2 10.0f

#define VIS1 10000.0f
#define VIS2 10000.0f

#define MU_INF 100.0f
#define MU_0   0.01f
#define ALPHA  1.0f
#define SCALING 100.0f
#define EXP_N  0.4f

#define MASS_TRESHOLD 0.005f
#define MASS_THESHOLD2 0.5f

#define LAUNCH_KERNEL(name, grid, block, ...)																				\
{																															\
	dim3 gridDims, blockDims;																								\
	computeGridSize3D(grid, block, gridDims, blockDims);																	\
	##name << < gridDims, blockDims>> >(__VA_ARGS__);																		\
	cudaError_t code = cudaDeviceSynchronize();																				\
	if (code != cudaSuccess)																								\
	{																														\
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); exit(code);					\
	}																														\
}
	static void computeGridSize3D(uint3 dims, uint3 blockSize, dim3& gridDim, dim3& blockDim)
	{
		gridDim.x = iDivUp(dims.x, blockSize.x);
		gridDim.y = iDivUp(dims.y, blockSize.y);
		gridDim.z = iDivUp(dims.z, blockSize.z);

		blockDim.x = blockSize.x;
		blockDim.y = blockSize.y;
		blockDim.z = blockSize.z;
	}

	struct Coef
	{
		float a;
		float x0;
		float x1;
		float y0;
		float y1;
		float z0;
		float z1;
	};

	template<typename TDataType>
	class PhaseFieldKernels
	{
		typedef typename TDataType::Real Real;

		typedef typename DArray3D<int> Grid1i;
		typedef typename DArray3D<Real> Grid1f;
		typedef typename DArray3D<Vector<Real, 3>> Grid3f;
		typedef typename DArray3D<Vector<Real, 4>> Grid4f;
		typedef typename DArray3D<Coef> GridCoef;

	public:
		PhaseFieldKernels() {};

		static void InterpolateVelocity(Grid3f vel, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w);

		static void InterpolateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid3f vel);

		static void AdvectBackward(Grid1f dst, Grid1f src, Grid3f vel, float dt);

		static void AdvectStaggeredBackward(Grid1f dst, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, float dt, int axis);

		static void AdvectForward(Grid1f dst, Grid1f src, Grid3f vel, float dt);

		static void AdvectBackward(Grid3f dst, Grid3f src, Grid3f vel, float dt);

		static void AdvectForward(Grid4f dst, Grid4f src, Grid3f vel, Grid1f weight, float dt);

		static void Sharpening(Grid1f dst, Grid3f dir, Grid1f src, Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f omega, float gamma,
			float h, float dt);

		static void Jacobi(Grid1f dst, Grid1f src, Grid1f buf, Grid3f vel, float a, float c, int iteration);

		static void ApplyViscosity(Grid3f dst, Grid3f src, Grid3f buf, Grid1f mass, float a, int iteration);

		static void PrepareForProjection(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, GridCoef coefMatrix, Grid1f RHS, Grid1f mass, float h, float dt);

		static void Projection(Grid1f pressure, Grid1f buf, GridCoef coefMatrix, Grid1f RHS, int numIter);

		static void UpdateVelocity(Grid1f vel_u, Grid1f vel_v, Grid1f vel_w, Grid1f pressure, Grid1f mass, float h, float dt);

		static void ApplyGravity(Grid3f v, Vec3f g, float dt);

		static void SetU(Grid1f vel_u);

		static void SetV(Grid1f vel_v);

		static void SetW(Grid1f vel_w);
	};
}