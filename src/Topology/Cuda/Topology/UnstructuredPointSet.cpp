#include "UnstructuredPointSet.h"

namespace dyno
{
	template<typename Coord>
	UnstructuredPointSet<Coord>::UnstructuredPointSet()
	{
	}

	template<typename Coord>
	UnstructuredPointSet<Coord>::~UnstructuredPointSet()
	{
	}

	template<typename TDataType>
	void UnstructuredPointSet<TDataType>::copyFrom(UnstructuredPointSet<TDataType>& pts)
	{
		mNeighborLists.assign(pts.mNeighborLists);

		PointSet<TDataType>::copyFrom(pts);
	}

	template<typename TDataType>
	DArrayList<int>& UnstructuredPointSet<TDataType>::getPointNeighbors()
	{
		return mNeighborLists;
	}

	template<typename TDataType>
	void UnstructuredPointSet<TDataType>::clear()
	{

	}

	DEFINE_CLASS(UnstructuredPointSet);
}