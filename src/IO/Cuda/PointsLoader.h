#pragma once
#include "GeometryLoader.h"

//Output
#include "Topology/PointSet.h"

namespace dyno
{
	/*!
	*	\class	PointsLoader
	*	\brief	Load a triangular mesh
	*/
	template<typename TDataType>
	class PointsLoader : public GeometryLoader<TDataType>
	{
		DECLARE_TCLASS(PointsLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointsLoader();
		virtual ~PointsLoader();

	public:
		DEF_INSTANCE_OUT(PointSet<TDataType>, PointSet, "");

	protected:
		void resetStates() override;
	};
}