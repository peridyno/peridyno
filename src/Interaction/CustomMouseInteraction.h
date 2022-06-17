#pragma once
#include "Module/InputMouseModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class CustomMouseIteraction : public InputMouseModule
	{
	DECLARE_TCLASS(CustomMouseIteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		CustomMouseIteraction();
		virtual ~CustomMouseIteraction() {};

		void calcIntersectClick();
		void calcIntersectDrag();

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, SelectedTriangleSet, "");
		DEF_INSTANCE_OUT(TriangleSet<TDataType>, OtherTriangleSet, "");

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

	IMPLEMENT_TCLASS(CustomMouseIteraction, TDataType)
}
