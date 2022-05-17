#pragma once
#include "Node.h"

#include "Topology/TriangleSet.h"
#include "Module/TopologyModule.h"
#include "Module/InputMouseModule.h"

namespace dyno
{
	template<typename TDataType> class MouseIntersect;

	template<typename TDataType>
	class MouseIntersect : public Node
	{
		DECLARE_TCLASS(MouseIntersect, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;

		MouseIntersect();
		~MouseIntersect() override;
		
		void calcIntersect();

		DEF_INSTANCE_IN(TRay3D<Real>, MouseRay, "");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, InitialTriangleSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, SelectedTriangleSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, OtherTriangleSet, "");

	protected:
		void resetStates() override;
		void updateStates() override;

	private:

	};

	IMPLEMENT_TCLASS(MouseIntersect, TDataType)
}