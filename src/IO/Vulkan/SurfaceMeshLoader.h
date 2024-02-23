#pragma once
#include "GeometryLoader.h"

//Output
#include "Topology/TriangleSet.h"

namespace dyno
{
	/*!
	*	\class	SurfaceMeshLoader
	*	\brief	Load a triangular mesh
	*/
	template<typename TDataType>
	class SurfaceMeshLoader : public GeometryLoader<TDataType>
	{
		DECLARE_TCLASS(SurfaceMeshLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		SurfaceMeshLoader();
		virtual ~SurfaceMeshLoader();

	public:
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "");

	protected:
		void resetStates() override;
	};
}