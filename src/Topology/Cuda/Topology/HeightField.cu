#include "HeightField.h"
#include <fstream>
#include <iostream>
#include <sstream>

#include "Math/Lerp.h"

namespace dyno
{
	IMPLEMENT_TCLASS(HeightField, TDataType)

	template<typename TDataType>
	HeightField<TDataType>::HeightField()
		: TopologyModule()
	{
		mDisplacement.resize(128, 128);
		mDisplacement.reset();

		mGridSpacing = Real(0.1);

		mOrigin = Coord(-mGridSpacing * 64, Real(0), -mGridSpacing * 64);
	}

	template<typename TDataType>
	HeightField<TDataType>::~HeightField()
	{
		mDisplacement.clear();
	}

	template<typename TDataType>
	void HeightField<TDataType>::setExtents(uint nx, uint ny)
	{
		mDisplacement.resize(nx, ny);
		mHeights.resize(nx, ny);

		mDisplacement.reset();
		mHeights.reset();
	}

	template<typename TDataType>
	uint HeightField<TDataType>::width()
	{
		return mDisplacement.nx();
	}

	template<typename TDataType>
	uint HeightField<TDataType>::height()
	{
		return mDisplacement.ny();
	}

	template<typename TDataType>
	void HeightField<TDataType>::copyFrom(HeightField<TDataType>& hf)
	{
		mOrigin = hf.mOrigin;
		mGridSpacing = hf.mGridSpacing;
		mDisplacement.assign(hf.mDisplacement);
		mHeights.assign(hf.mHeights);
	}

	template <typename Real, typename Coord>
	__global__ void PS_Scale(
		DArray<Coord> vertex,
		Real s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;
		//return;
		vertex[pId] = vertex[pId] * s;
	}

	template<typename TDataType>
	void HeightField<TDataType>::scale(Real s)
	{
		//cuExecute(m_coords.size(), PS_Scale, m_coords, s);
	}

	template <typename Coord>
	__global__ void PS_Scale(
		DArray<Coord> vertex,
		Coord s)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		Coord pos_i = vertex[pId];
		vertex[pId] = Coord(pos_i[0] * s[0], pos_i[1] * s[1], pos_i[2] * s[2]);
	}

	template<typename TDataType>
	void HeightField<TDataType>::scale(Coord s)
	{
		//cuExecute(m_coords.size(), PS_Scale, m_coords, s);
	}

	template <typename Coord>
	__global__ void PS_Translate(
		DArray<Coord> vertex,
		Coord t)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= vertex.size()) return;

		vertex[pId] = vertex[pId] + t;
	}


	template<typename TDataType>
	void HeightField<TDataType>::translate(Coord t)
	{
		//cuExecute(m_coords.size(), PS_Translate, m_coords, t);

// 		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);
// 
// 		PS_Translate << <pDims, BLOCK_SIZE >> > (
// 			m_coords,
// 			t);
// 		cuSynchronize();
	}

	//Back tracing using the semi-Lagrangian scheme
	template <typename Real, typename Coord>
	__global__ void HF_RasterizeDisplacements(
		DArray2D<Real> vertical,
		DArray2D<Coord> displacements,
		Coord origin,
		Real h)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

		uint nx = displacements.nx();
		uint ny = displacements.ny();

		if (i < nx && j < ny)
		{
			Coord disp_ij = displacements(i, j);

			Coord interp_ij = bilinear(displacements, i - disp_ij.x / h, j - disp_ij.z / h, LerpMode::REPEAT);

			vertical(i, j) = interp_ij.y;
		}
	}

	template<typename TDataType>
	DArray2D<typename TDataType::Real>& HeightField<TDataType>::calculateHeightField()
	{
		uint nx = mDisplacement.nx();
		uint ny = mDisplacement.ny();

		if (nx != mHeights.nx() || ny != mHeights.ny()) {
			mHeights.resize(nx, ny);
		}
		
		cuExecute2D(make_uint2(nx, ny),
			HF_RasterizeDisplacements,
			mHeights,
			mDisplacement,
			mOrigin,
			mGridSpacing);
		
		return mHeights;
	}

	DEFINE_CLASS(HeightField);
}