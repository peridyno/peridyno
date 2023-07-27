#include "HeightField.h"
#include <fstream>
#include <iostream>
#include <sstream>

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
		mDisplacement.reset();
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

	DEFINE_CLASS(HeightField);
}