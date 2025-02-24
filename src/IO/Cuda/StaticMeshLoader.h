#pragma once
#include "Node/ParametricModel.h"

#include "Topology/TriangleSet.h"
#include "Field/FilePath.h"

namespace dyno
{
	template <typename TDataType> class TriangleSet;
	/*!
	*	\class	StaticMeshLoader
	*	\brief	A node containing a TriangleSet object
	*
	*	This class is typically used as a representation for a static boundary mesh or base class to other classes
	*
	*/
	template<typename TDataType>
	class StaticMeshLoader : public ParametricModel<TDataType>
	{
		DECLARE_TCLASS(StaticMeshLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		StaticMeshLoader();
		std::string getNodeType() override { return "IO"; }
	public:
		DEF_VAR(FilePath, FileName, "", "");
		
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, InitialTriangleSet, "Initial topology");
		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "Transformed Topology");
	};
}