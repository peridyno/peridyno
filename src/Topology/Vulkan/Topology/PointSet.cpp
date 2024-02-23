#include "PointSet.h"

namespace dyno
{
	template<typename TDataType>
	PointSet<TDataType>::PointSet()
		: TopologyModule()
	{
	}

	template<typename TDataType>
	PointSet<TDataType>::~PointSet()
	{
	}

	template<typename TDataType>
	void PointSet<TDataType>::setPoints(std::vector<Vec3f>& points)
	{
		mCoords.assign(points);
	}

	template<typename TDataType>
	void PointSet<TDataType>::setPoints(const DArray<Vec3f>& points)
	{
		mCoords.assign(points);


	}

	template<typename TDataType>
	bool PointSet<TDataType>::isEmpty()
	{
		return mCoords.size() == 0;
	}

	template<typename TDataType>
	void PointSet<TDataType>::clear()
	{
		mCoords.clear();
	}

	DEFINE_CLASS(PointSet)
}