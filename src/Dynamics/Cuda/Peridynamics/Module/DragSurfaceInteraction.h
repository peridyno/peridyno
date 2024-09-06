#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"

#include "Collision/Attribute.h"

/**
 * @brief This is an implementation of drag interaction calculated by triangles.
 *
 *	This algorithm simply take mouse coordinate on screen, while transforming it into intersection plane in world space.
 *  The selected point (surface vertex) will shift to fix through directed manipulation on attribute, position and velocity of particles,
 *	and the attribute will be restored when mouse-press released. 
 *
 *
 */

namespace dyno
{
	template<typename TDataType>
	class DragSurfaceInteraction : public MouseInputModule
	{
		DECLARE_TCLASS(DragSurfaceInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Triangle Triangle;

		DragSurfaceInteraction();
		virtual ~DragSurfaceInteraction() {};

		void InteractionClick();
		void InteractionDrag();
		void calcSurfaceInteractClick();
		void calcSurfaceInteractDrag();
		void setTriFixed();
		void cancelVelocity();

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_VAR(Real, InterationRadius, 0.005, "The radius of interaction");
	
		DEF_VAR_IN(Real, TimeStep, "Time step");


	protected:

		void onEvent(PMouseEvent event) override;

	private:
		bool needInit = true;
		std::shared_ptr<Camera> camera;
		TRay3D<Real> ray1;
		TRay3D<Real> ray2;
		Real x1;
		Real y1;
		Real x2;
		Real y2;
		bool isPressed;

		DArray<int> triIntersectedIndex; //reference for all triangles, int: 0,1
		DArray<int> intersected_triangles; // reference for intersected list, int: tId
		DArray<Attribute> restoreAttribute; 
		int intersectionCenterIndex;  //intersected triangle (with out radius expanded).
		DArray<Coord> intersectionCenter; //current pos of intersected triangle[intersectionCenterIndex]
	};

	IMPLEMENT_TCLASS(DragSurfaceInteraction, TDataType)
}
