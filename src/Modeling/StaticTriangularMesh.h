#pragma once
#include "Node/ParametricModel.h"

#include "Topology/TriangleSet.h"
#include "FilePath.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	StaticTriangularMesh
	*	\brief	A node containing a TriangleSet object
	*
	*	This class is typically used as a representation for a static boundary mesh or base class to other classes
	*
	*/
	template<typename TDataType>
	class StaticTriangularMesh : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(StaticTriangularMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticTriangularMesh();

	public:
		DEF_VAR(FilePath, FileName, "", "");
		
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InitialTopology, "Initial topology");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, Topology, "Transformed Topology");
	};
}