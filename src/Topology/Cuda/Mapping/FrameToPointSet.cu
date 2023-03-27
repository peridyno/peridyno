#pragma once
#include "FrameToPointSet.h"
#include "Topology/Frame.h"
#include "Topology/PointSet.h"

namespace dyno
{
	template<typename TDataType>
	FrameToPointSet<TDataType>::FrameToPointSet()
		: TopologyMapping()
	{

	}

	template<typename TDataType>
	FrameToPointSet<TDataType>::FrameToPointSet(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
		: TopologyMapping()
	{
		m_from = from;
		m_to = to;
	}


	template<typename TDataType>
	FrameToPointSet<TDataType>::~FrameToPointSet()
	{
		if (m_refPoints.begin() != NULL)
		{
			m_refPoints.clear();
		}
	}


	template<typename TDataType>
	void FrameToPointSet<TDataType>::initialize(const Rigid& rigid, DArray<Coord>& points)
	{
		m_refRigid = rigid;
		m_refPoints.resize(points.size());
		m_refPoints.assign(points);
	}


	template<typename TDataType>
	void FrameToPointSet<TDataType>::match(std::shared_ptr<Frame<TDataType>> from, std::shared_ptr<PointSet<TDataType>> to)
	{
		m_initFrom = std::make_shared<Frame<TDataType>>();
		m_initTo = std::make_shared<PointSet<TDataType>>();

		m_initFrom->copyFrom(*from);
		m_initTo->copyFrom(*to);
	}


	template <typename Coord, typename Rigid, typename Matrix>
	__global__ void ApplyRigidTranform(
		DArray<Coord> points,
		Coord curCenter,
		Matrix curMat,
		DArray<Coord> refPoints,
		Coord refCenter,
		Matrix refMat)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;

		points[pId] = curCenter + curMat*refMat.transpose()*(refPoints[pId] - refCenter);
	}

	template<typename TDataType>
	void FrameToPointSet<TDataType>::applyTransform(const Rigid& rigid, DArray<Coord>& points)
	{
		if (points.size() != m_refPoints.size())
		{
			std::cout << "The array sizes does not match for RigidToPoints" << std::endl;
		}

		uint pDims = cudaGridSize(points.size(), BLOCK_SIZE);

		ApplyRigidTranform<Coord, Rigid, Matrix><< <pDims, BLOCK_SIZE >> >(points, rigid.getCenter(), rigid.getRotationMatrix(), m_refPoints, m_refRigid.getCenter(), m_refRigid.getRotationMatrix());
	}

	template<typename TDataType>
	bool FrameToPointSet<TDataType>::apply()
	{
		DArray<Coord>& m_coords = m_initTo->getPoints();

		uint pDims = cudaGridSize(m_coords.size(), BLOCK_SIZE);

		ApplyRigidTranform<Coord, Rigid, Matrix> << <pDims, BLOCK_SIZE >> >(
			m_to->getPoints(),
			m_from->getCenter(), 
			m_from->getOrientation(),
			m_coords,
			m_initFrom->getCenter(), 
			m_initFrom->getOrientation());

		return true;
	}


	template<typename TDataType>
	bool FrameToPointSet<TDataType>::initializeImpl()
	{
		match(m_from, m_to);
		return true;
	}

	DEFINE_CLASS(FrameToPointSet);
}