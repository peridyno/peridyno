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

	template<typename Real>
	__device__ Real D_Weight(Real r, Real h)
	{
		CorrectedKernel<Real> kernSmooth;
		return kernSmooth.WeightRR(r, 2*h);
// 		h = h < EPSILON ? Real(1) : h;
// 		return 1 / (h*h*h);
	}


	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_PrecomputeShape(
		DArray<Matrix> invK,
		DArrayList<NPair> restShapes)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= invK.size()) return;

		List<NPair>& restShape_i = restShapes[pId];
		NPair np_i = restShape_i[0];
		Coord rest_i = np_i.pos;
		int size_i = restShape_i.size();
		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_i - rest_pos_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;
		Real smoothingLength = maxDist;

		Real total_weight = 0.0f;
		Matrix mat_i = Matrix(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_j = np_j.pos;
			Real r = (rest_i - rest_j).norm();

			if (r > EPSILON)
			{
				Real weight = D_Weight(r, smoothingLength);
				Coord q = (rest_j - rest_i) / smoothingLength*sqrt(weight);

				mat_i(0, 0) += q[0] * q[0]; mat_i(0, 1) += q[0] * q[1]; mat_i(0, 2) += q[0] * q[2];
				mat_i(1, 0) += q[1] * q[0]; mat_i(1, 1) += q[1] * q[1]; mat_i(1, 2) += q[1] * q[2];
				mat_i(2, 0) += q[2] * q[0]; mat_i(2, 1) += q[2] * q[1]; mat_i(2, 2) += q[2] * q[2];

				total_weight += weight;
			}
		}

		if (total_weight > EPSILON)
		{
			mat_i *= (1.0f / total_weight);
		}

		Matrix R(0), U(0), D(0), V(0);

// 		if (pId == 0)
// 		{
// 			printf("EM_PrecomputeShape**************************************");
// 
// 			printf("K: \n %f %f %f \n %f %f %f \n %f %f %f \n\n\n",
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2));
// 		}

		polarDecomposition(mat_i, R, U, D, V);

		if (mat_i.determinant() < EPSILON*smoothingLength)
		{
			Real threshold = 0.0001f*smoothingLength;
			D(0, 0) = D(0, 0) > threshold ? 1.0 / D(0, 0) : 1.0;
			D(1, 1) = D(1, 1) > threshold ? 1.0 / D(1, 1) : 1.0;
			D(2, 2) = D(2, 2) > threshold ? 1.0 / D(2, 2) : 1.0;

			mat_i = V * D*U.transpose();
		}
		else
			mat_i = mat_i.inverse();

		invK[pId] = mat_i;
	}

	template <typename Real, typename Coord, typename Matrix, typename NPair>
	__global__ void EM_EnforceElasticity(
		DArray<Coord> delta_position,
		DArray<Real> weights,
		DArray<Real> bulkCoefs,
		DArray<Matrix> invK,
		DArray<Coord> position,
		DArrayList<NPair> restShapes,
		Real mu,
		Real lambda)
	{

		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		List<NPair>& restShape_i = restShapes[pId];
		NPair np_i = restShape_i[0];
		Coord rest_i = np_i.pos;
		int size_i = restShape_i.size();

		Coord cur_pos_i = position[pId];

		Coord accPos = Coord(0);
		Real accA = Real(0);
		Real bulk_i = bulkCoefs[pId];

		Real maxDist = Real(0);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_pos_j = np_j.pos;
			Real r = (rest_i - rest_pos_j).norm();

			maxDist = max(maxDist, r);
		}
		maxDist = maxDist < EPSILON ? Real(1) : maxDist;
		Real horizon = maxDist;


		Real total_weight = 0.0f;
		Matrix deform_i = Matrix(0.0f);
		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Real r = (rest_j - rest_i).norm();

			if (r > EPSILON)
			{
				Real weight = D_Weight(r, horizon);

				Coord p = (position[j] - position[pId]) / horizon;
				Coord q = (rest_j - rest_i) / horizon*weight;

				deform_i(0, 0) += p[0] * q[0]; deform_i(0, 1) += p[0] * q[1]; deform_i(0, 2) += p[0] * q[2];
				deform_i(1, 0) += p[1] * q[0]; deform_i(1, 1) += p[1] * q[1]; deform_i(1, 2) += p[1] * q[2];
				deform_i(2, 0) += p[2] * q[0]; deform_i(2, 1) += p[2] * q[1]; deform_i(2, 2) += p[2] * q[2];
				total_weight += weight;
			}
		}


		if (total_weight > EPSILON)
		{
			deform_i *= (1.0f / total_weight);
			deform_i = deform_i * invK[pId];
		}
		else
		{
			total_weight = 1.0f;
		}

		//Check whether the reference shape is inverted, if yes, simply set K^{-1} to be an identity matrix
		//Note other solutions are possible.
		if ((deform_i.determinant()) < -0.001f)
		{
			deform_i = Matrix::identityMatrix(); 
		}


// 		//if (pId == 0)
// 		{
// 			Matrix mat_i = invK[pId];
// 			printf("Mat %d: \n %f %f %f \n %f %f %f \n %f %f %f \n", 
// 				pId,
// 				mat_i(0, 0), mat_i(0, 1), mat_i(0, 2),
// 				mat_i(1, 0), mat_i(1, 1), mat_i(1, 2),
// 				mat_i(2, 0), mat_i(2, 1), mat_i(2, 2));
// 		}

		for (int ne = 0; ne < size_i; ne++)
		{
			NPair np_j = restShape_i[ne];
			Coord rest_j = np_j.pos;
			int j = np_j.index;

			Coord cur_pos_j = position[j];
			Real r = (rest_j - rest_i).norm();

			if (r > 0.01f*horizon)
			{
				Real weight = D_Weight(r, horizon);

				Coord rest_dir_ij = deform_i*(rest_i - rest_j);
				Coord cur_dir_ij = cur_pos_i - cur_pos_j;

				cur_dir_ij = cur_dir_ij.norm() > EPSILON ? cur_dir_ij.normalize() : Coord(0);
				rest_dir_ij = rest_dir_ij.norm() > EPSILON ? rest_dir_ij.normalize() : Coord(0, 0, 0);

				Real mu_ij = mu*bulk_i* D_Weight(r, horizon);
				Coord mu_pos_ij = position[j] + r*rest_dir_ij;
				Coord mu_pos_ji = position[pId] - r*rest_dir_ij;

				Real lambda_ij = lambda*bulk_i*D_Weight(r, horizon);
				Coord lambda_pos_ij = position[j] + r*cur_dir_ij;
				Coord lambda_pos_ji = position[pId] - r*cur_dir_ij;

				Coord delta_pos_ij = mu_ij*mu_pos_ij + lambda_ij*lambda_pos_ij;
				Real delta_weight_ij = mu_ij + lambda_ij;

				Coord delta_pos_ji = mu_ij*mu_pos_ji + lambda_ij*lambda_pos_ji;

				accA += delta_weight_ij;
				accPos += delta_pos_ij;


				atomicAdd(&weights[j], delta_weight_ij);
				atomicAdd(&delta_position[j][0], delta_pos_ji[0]);
				atomicAdd(&delta_position[j][1], delta_pos_ji[1]);
				atomicAdd(&delta_position[j][2], delta_pos_ji[2]);
			}
		}

		atomicAdd(&weights[pId], accA);
		atomicAdd(&delta_position[pId][0], accPos[0]);
		atomicAdd(&delta_position[pId][1], accPos[1]);
		atomicAdd(&delta_position[pId][2], accPos[2]);
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition(
		DArray<Coord> position,
		DArray<Coord> old_position,
		DArray<Coord> delta_position,
		DArray<Real> delta_weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		position[pId] = (old_position[pId] + delta_position[pId]) / (1.0+delta_weights[pId]);
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdatePosition2(
		DArray<Coord> position,
		DArray<Coord> old_position,
		DArray<Coord> delta_position,
		DArray<Real> delta_weights,
		float4* height)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		//printf("%d %f %f %d\n ", pId, position[pId], height[pId].x, position.size());
		//printf("%d\n ", position.size());
		//position[pId] = (old_position[pId] + delta_position[pId]) / (1.0 + delta_weights[pId]);
		position[pId] = Coord(pId%10);
	}

	template <typename Real, typename Coord>
	__global__ void K_UpdateVelocity(
		DArray<Coord> velArr,
		DArray<Coord> prePos,
		DArray<Coord> curPos,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velArr.size()) return;

		velArr[pId] += (curPos[pId] - prePos[pId]) / dt;
	}

	__constant__ float GRAVITY = 9.83219f * 0.5f; //0.5f * Fallbeschleunigung
	template<typename TDataType>
	CapillaryWaveModule<TDataType>::CapillaryWaveModule()
		: ConstraintModule()
	{
		this->inHorizon()->setValue(0.0125);
		this->inNeighborIds()->tagOptional(true);

		int size = 512;
		float patchLength = 512.0;

		m_patch_length = patchLength;
		m_realGridSize = patchLength / size;

		m_simulatedRegionWidth = size;
		m_simulatedRegionHeight = size;

		m_simulatedOriginX = 0;
		m_simulatedOriginY = 0;

		initialize();
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::initSource()
	{
		int sizeInBytes = m_simulatedRegionWidth * m_simulatedRegionHeight * sizeof(float2);

		cudaCheck(cudaMalloc(&m_source, sizeInBytes));
		cudaCheck(cudaMalloc(&m_weight, m_simulatedRegionWidth * m_simulatedRegionHeight * sizeof(float)));
		cudaCheck(cudaMemset(m_source, 0, sizeInBytes));

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
	void CapillaryWaveModule<TDataType>::initialize()
	{
		initDynamicRegion();

		initSource();
	}

	__global__ void C_InitDynamicRegion(gridpoint* grid, int gridwidth, int gridheight, int pitch, float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			gridpoint gp;
			gp.x = level;
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = 0.0f;

			grid2Dwrite(grid, x, y, pitch, gp);
		}
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::initDynamicRegion()
	{
		


		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		size_t pitch;
		cudaCheck(cudaMallocPitch(&m_device_grid, &pitch, extNx * sizeof(gridpoint), extNy));
		cudaCheck(cudaMallocPitch(&m_device_grid_next, &pitch, extNx * sizeof(gridpoint), extNy));

		cudaCheck(cudaMalloc((void**)&m_height, m_simulatedRegionWidth * m_simulatedRegionWidth * sizeof(float4)));

		//gl_utility::createTexture(m_simulatedRegionWidth, m_simulatedRegionHeight, GL_RGBA32F, m_height_texture, GL_CLAMP_TO_BORDER, GL_LINEAR, GL_LINEAR, GL_RGBA, GL_FLOAT);
		//cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_texture, m_height_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

		m_grid_pitch = pitch / sizeof(gridpoint);

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

		//error = cudaThreadSynchronize();

		//g_cpChannelDesc = cudaCreateChannelDesc<float4>();
		//cudaCheck(cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint)));
		
	}

	template<typename TDataType>
	CapillaryWaveModule<TDataType>::~CapillaryWaveModule()
	{
		mWeights.clear();
		mDisplacement.clear();
		mInvK.clear();
		mF.clear();
		mPosBuf.clear();
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::enforceElasticity()
	{
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		mDisplacement.reset();
		mWeights.reset();
/*
		EM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
			mDisplacement,
			mWeights,
			mBulkStiffness,
			mInvK,
			this->inPosition()->getData(),
			this->inRestShape()->getData(),
			this->varMu()->getData(),
			this->varLambda()->getData());
		cuSynchronize();

		K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getData(),
			mPosBuf,
			mDisplacement,
			mWeights);
		cuSynchronize();
*/
		K_UpdatePosition2 << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getData(),
			mPosBuf,
			mDisplacement,
			mWeights,
			m_height);
		cuSynchronize();
	}

	template<typename Real>
	__global__ void EM_InitBulkStiffness(DArray<Real> stiffness)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= stiffness.size()) return;

		stiffness[pId] = Real(1);
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::computeMaterialStiffness()
	{
	
		int num = this->inPosition()->getElementCount();

		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		EM_InitBulkStiffness << <pDims, BLOCK_SIZE >> > (mBulkStiffness);
	}


	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::computeInverseK()
	{
	
		auto& restShapes = this->inRestShape()->getData();
		uint pDims = cudaGridSize(restShapes.size(), BLOCK_SIZE);

		EM_PrecomputeShape <Real, Coord, Matrix, NPair> << <pDims, BLOCK_SIZE >> > (
			mInvK,
			restShapes);
		cuSynchronize();
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::solveElasticity()
	{

		//Save new positions
		mPosBuf.assign(this->inPosition()->getData());

		this->computeInverseK();

		int itor = 0;
		uint maxIterNum = this->varIterationNumber()->getData();
		while (itor < maxIterNum) {
			this->enforceElasticity();
			itor++;
		}

		//this->updateVelocity();
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::updateVelocity()
	{
		
		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		
		Real dt = this->inTimeStep()->getData();
		
		K_UpdateVelocity << <pDims, BLOCK_SIZE >> > (
			this->inVelocity()->getData(),
			mPosBuf,
			this->inPosition()->getData(),
			dt);
		cuSynchronize();
	}


	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::constrain()
	{

		int num = this->inPosition()->getElementCount();
		uint pDims = cudaGridSize(num, BLOCK_SIZE);

		mDisplacement.reset();
		mWeights.reset();

		K_UpdatePosition2 << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getData(),
			mPosBuf,
			mDisplacement,
			mWeights,
			m_height);
		cuSynchronize();

		float dt = 0.016f;
		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

	
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

		C_InitHeightField << < blocksPerGrid, threadsPerBlock >> > (m_height, m_device_grid,  m_simulatedRegionWidth, m_horizon, m_realGridSize);
		synchronCheck;
		C_InitHeightGrad << < blocksPerGrid, threadsPerBlock >> > (m_height, m_simulatedRegionWidth);
		synchronCheck;




		/*
				EM_EnforceElasticity << <pDims, BLOCK_SIZE >> > (
					mDisplacement,
					mWeights,
					mBulkStiffness,
					mInvK,
					this->inPosition()->getData(),
					this->inRestShape()->getData(),
					this->varMu()->getData(),
					this->varLambda()->getData());
				cuSynchronize();

				K_UpdatePosition << <pDims, BLOCK_SIZE >> > (
					this->inPosition()->getData(),
					mPosBuf,
					mDisplacement,
					mWeights);
				cuSynchronize();
		*/
		K_UpdatePosition2 << <pDims, BLOCK_SIZE >> > (
			this->inPosition()->getData(),
			mPosBuf,
			mDisplacement,
			mWeights,
			m_height);
		cuSynchronize();
		//this->solveElasticity();
	}


	template <typename Coord, typename NPair>
	__global__ void K_UpdateRestShape(
		DArrayList<NPair> shape,
		DArrayList<int> nbr,
		DArray<Coord> pos)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		NPair np;

		List<NPair>& rest_shape_i = shape[pId];
		List<int>& list_id_i = nbr[pId];
		int nbSize = list_id_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_id_i[ne];
			np.index = j;
			np.pos = pos[j];
			np.weight = 1;

			rest_shape_i.insert(np);
			if (pId == j)
			{
				NPair np_0 = rest_shape_i[0];
				rest_shape_i[0] = np;
				rest_shape_i[ne] = np_0;
			}
		}
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::swapDeviceGrid()
	{
		gridpoint* grid_helper = m_device_grid;
		m_device_grid = m_device_grid_next;
		m_device_grid_next = grid_helper;
	}

	template<typename TDataType>
	void CapillaryWaveModule<TDataType>::preprocess()
	{
		int num = this->inPosition()->getElementCount();

		if (num == mInvK.size())
			return;

		mInvK.resize(num);
		mWeights.resize(num);
		mDisplacement.resize(num);

		mF.resize(num);

		mPosBuf.resize(num);
		mBulkStiffness.resize(num);

		this->computeMaterialStiffness();
	}

	__global__ void C_InitSource(
		float2* source,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			if (i < patchSize / 2 + 3 && i > patchSize / 2 - 3 && j < patchSize / 2 + 3 && j > patchSize / 2 - 3)
			{
				float2 uv;
				uv.x = 1.0f;
				uv.y = 1.0f;
				source[i + j * patchSize] = uv;
			}
		}
	}

	__global__ void C_ImposeBC(float4* grid_next, float4* grid, int width, int height, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			if (x == 0)
			{
				float4 a = grid2Dread(grid, 1, y, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (x == width - 1)
			{
				float4 a = grid2Dread(grid, width - 2, y, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (y == 0)
			{
				float4 a = grid2Dread(grid, x, 1, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else if (y == height - 1)
			{
				float4 a = grid2Dread(grid, x, height - 2, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
			else
			{
				float4 a = grid2Dread(grid, x, y, pitch);
				grid2Dwrite(grid_next, x, y, pitch, a);
			}
		}
	}

	__host__ __device__ void C_FixShore(gridpoint& l, gridpoint& c, gridpoint& r)
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

	__host__ __device__ gridpoint C_VerticalPotential(gridpoint gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));

		gridpoint G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * h * h;
		G.w = 0.0f;
		return G;
	}

	__device__ gridpoint C_HorizontalPotential(gridpoint gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		gridpoint F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * h * h;
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}

	__device__ gridpoint C_SlopeForce(gridpoint c, gridpoint n, gridpoint e, gridpoint s, gridpoint w)
	{
		float h = max(c.x, 0.0f);

		gridpoint H;
		H.x = 0.0f;
		H.y = -GRAVITY * h * (e.w - w.w);
		H.z = -GRAVITY * h * (s.w - n.w);
		H.w = 0.0f;
		return H;
	}

	__global__ void C_OneWaveStep(gridpoint* grid_next, gridpoint* device_grid, int width, int height, float timestep, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			gridpoint center = grid2Dread(device_grid, gridx, gridy, pitch);

			gridpoint north = grid2Dread(device_grid, gridx, gridy - 1, pitch);

			gridpoint west = grid2Dread(device_grid, gridx - 1, gridy, pitch);

			gridpoint south = grid2Dread(device_grid, gridx, gridy + 1, pitch);

			gridpoint east = grid2Dread(device_grid, gridx + 1, gridy, pitch);

			C_FixShore(west, center, east);
			C_FixShore(north, center, south);

			gridpoint u_south = 0.5f * (south + center) - timestep * (C_VerticalPotential(south) - C_VerticalPotential(center));
			gridpoint u_north = 0.5f * (north + center) - timestep * (C_VerticalPotential(center) - C_VerticalPotential(north));
			gridpoint u_west = 0.5f * (west + center) - timestep * (C_HorizontalPotential(center) - C_HorizontalPotential(west));
			gridpoint u_east = 0.5f * (east + center) - timestep * (C_HorizontalPotential(east) - C_HorizontalPotential(center));

			gridpoint u_center = center + timestep * C_SlopeForce(center, north, east, south, west) - timestep * (C_HorizontalPotential(u_east) - C_HorizontalPotential(u_west)) - timestep * (C_VerticalPotential(u_south) - C_VerticalPotential(u_north));
			u_center.x = max(0.0f, u_center.x);

			grid2Dwrite(grid_next, gridx, gridy, pitch, u_center);
		}
	}

	__global__ void C_InitHeightField(
		float4* height,
		gridpoint* device_grid,
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

			gridpoint gp = grid2Dread(device_grid, gridx, gridy, patchSize);
			height[i + j * patchSize].x = gp.x - horizon;

			float d = sqrtf((i - patchSize / 2) * (i - patchSize / 2) + (j - patchSize / 2) * (j - patchSize / 2));
			float q = d / (0.49f * patchSize);

			float weight = q < 1.0f ? 1.0f - q * q : 0.0f;
			height[i + j * patchSize].y = 1.3f * realSize * sinf(3.0f * weight * height[i + j * patchSize].x * 0.5f * M_PI);

			// x component stores the original height, y component stores the normalized height, z component stores the X gradient, w component stores the Z gradient;
		}
	}


	__global__ void C_InitHeightGrad(
		float4* height,
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

			float4 Dx = (height[i_plus_one + j * patchSize] - height[i_minus_one + j * patchSize]) / 2;
			float4 Dz = (height[i + j_plus_one * patchSize] - height[i + j_minus_one * patchSize]) / 2;

			height[i + patchSize * j].z = Dx.y;
			height[i + patchSize * j].w = Dz.y;
		}
	}
	DEFINE_CLASS(CapillaryWaveModule);
}