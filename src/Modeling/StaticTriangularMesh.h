#pragma once
#include "Node.h"
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
	class StaticTriangularMesh : public Node
	{
		DECLARE_TCLASS(StaticTriangularMesh, TDataType)
	public:
		StaticTriangularMesh();

	public:
		DEF_VAR(Vec3f, Location, 0, "Node location");
		DEF_VAR(Vec3f, Rotation, 0, "Node rotation");
		DEF_VAR(Vec3f, Scale, Vec3f(1.0f), "Node scale");

		DEF_VAR(FilePath, FileName, "", "");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

	protected:
		void resetStates() override;
	};
}