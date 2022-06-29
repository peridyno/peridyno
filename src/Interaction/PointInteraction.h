#pragma once
#include "Module/InputMouseModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class PointIteraction : public InputMouseModule
	{
		DECLARE_TCLASS(PointIteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PointIteraction();
		virtual ~PointIteraction() {};

		void calcIntersectClick();
		void calcIntersectDrag();

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");
		DEF_INSTANCE_OUT(PointSet<TDataType>, SelectedPointSet, "");
		DEF_INSTANCE_OUT(PointSet<TDataType>, OtherPointSet, "");
		DEF_VAR(Real, InterationRadius, 0.01, "The radius of interaction");

	protected:
		void onEvent(PMouseEvent event) override;
	private:
		std::shared_ptr<Camera> camera;
		TRay3D<Real> ray1;
		TRay3D<Real> ray2;
		Real x1;
		Real y1;
		Real x2;
		Real y2;
		bool isPressed;
	};

	IMPLEMENT_TCLASS(PointIteraction, TDataType)
}
