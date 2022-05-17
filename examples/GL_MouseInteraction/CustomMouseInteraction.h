#pragma once
#include "Module/InputMouseModule.h"
#include "Module/TopologyModule.h"
#include "Module/MouseIntersect.h"

namespace dyno
{
	class CustomMouseIteraction : public InputMouseModule
	{
	public:
		CustomMouseIteraction() {};
		virtual ~CustomMouseIteraction() {};

		DEF_INSTANCE_IN(TopologyModule, Topology, "");
		DEF_INSTANCE_STATE(MouseIntersect<DataType3f>, MouseIntersect, "");
		DEF_INSTANCE_STATE(TRay3D<Real>, MouseRay, "");

	protected:
		void onEvent(PMouseEvent event) override;
	};
}
