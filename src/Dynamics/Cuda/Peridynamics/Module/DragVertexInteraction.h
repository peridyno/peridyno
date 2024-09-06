#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"
#include "Topology/TriangleSet.h"
#include "Collision/Attribute.h"

/**
 * @brief This is an implementation of drag interaction calculated by vertex, with topo of triangularMesh.
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
	class DragVertexInteraction : public MouseInputModule
	{
		DECLARE_TCLASS(DragVertexInteraction, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;
		typedef typename TopologyModule::Triangle Triangle;

		DragVertexInteraction();
		virtual ~DragVertexInteraction() {};

		void InteractionClick();
		void InteractionDrag();
		void calcVertexInteractClick();
		void calcVertexInteractDrag();
		void setVertexFixed();
		void cancelVelocity();

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");

		DEF_ARRAY_IN(Attribute, Attribute, DeviceType::GPU, "Particle attribute");
		
		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");

		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");

		DEF_VAR(Real, InterationRadius, 0.03, "The radius of interaction");
	
		DEF_VAR_IN(Real, TimeStep, "Time step");


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
		bool needInit = true;
		DArray<int> verIntersectedIndex; //reference for all vertex, int: 0,1
		DArray<int> intersected_vertex; // reference for intersected list, int: vId
		DArray<Attribute> restoreAttribute; 
		DArray<Coord> intersectionCenter; //current pos of intersected vertex[intersectionCenterIndex]
	};

	IMPLEMENT_TCLASS(DragVertexInteraction, TDataType)
}
