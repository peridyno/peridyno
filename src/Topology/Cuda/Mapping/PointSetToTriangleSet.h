#pragma once
#include "Node.h"

#include "Topology/PointSet.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType> class PointSetToPointSet;

	template<typename TDataType>
	class PointSetToTriangleSet : public Node
	{
		DECLARE_TCLASS(PointSetToTriangleSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointSetToTriangleSet();
		PointSetToTriangleSet(Real r) : mRadius(r) {};
		~PointSetToTriangleSet() override;

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");

 		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialShape, "");
 
 		DEF_INSTANCE_OUT(TriangleSet<TDataType>, Shape, "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:
		std::shared_ptr<PointSetToPointSet<TDataType>> mPointMapper;
		Real mRadius = 0.0125;
	};

	IMPLEMENT_TCLASS(PointSetToTriangleSet, TDataType)
}