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
	}

	template<typename TDataType>
	HeightField<TDataType>::~HeightField()
	{
	}

	template<typename TDataType>
	void HeightField<TDataType>::setSpace(Real dx, Real dz)
	{
		m_dx = dx;
		m_dz = dz;
	}

	template<typename TDataType>
	void HeightField<TDataType>::copyFrom(HeightField<TDataType>& pointSet)
	{
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