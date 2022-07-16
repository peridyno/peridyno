#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	class CustomMouseIteraction : public MouseInputModule
	{
	public:
		CustomMouseIteraction() {};
		virtual ~CustomMouseIteraction() {};

		DEF_INSTANCE_IN(TopologyModule, Topology, "");

	protected:
		void onEvent(PMouseEvent event) override;
	};
}
