#pragma once
#include "Node.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	/*!
	*	\class	SurfaceMeshLoader
	*	\brief	Load a triangular mesh
	*/
	template<typename TDataType>
	class SurfaceMeshLoader : public Node
	{
		DECLARE_TCLASS(SurfaceMeshLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceMeshLoader();
		virtual ~SurfaceMeshLoader();

		DEF_VAR(std::string, FileName, "", "");

	public:
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override;
	};
}