#include "SimplexSet.h"

namespace dyno
{
	template<typename TDataType>
	SimplexSet<TDataType>::SimplexSet()
		: PointSet<TDataType>()
	{
	}

	template<typename TDataType>
	SimplexSet<TDataType>::~SimplexSet()
	{
		mSegmentIndex.clear();
		mTriangleIndex.clear();
		mTetrahedronSet.clear();
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::copyFrom(SimplexSet<TDataType>& simplex)
	{
		PointSet<TDataType>::copyFrom(simplex);

		mSegmentIndex.assign(simplex.mSegmentIndex);
		mTriangleIndex.assign(simplex.mTriangleIndex);
		mTetrahedronSet.assign(simplex.mTetrahedronSet);
	}

	template<typename TDataType>
	bool SimplexSet<TDataType>::isEmpty()
	{
		bool empty = true;
		empty |= mSegmentIndex.size() == 0;
		empty |= mTriangleIndex.size() == 0;
		empty |= mTetrahedronSet.size() == 0;

		return empty;
	}

	template<typename TDataType>
	void SimplexSet<TDataType>::updateTopology()
	{

	}

	DEFINE_CLASS(SimplexSet);
}