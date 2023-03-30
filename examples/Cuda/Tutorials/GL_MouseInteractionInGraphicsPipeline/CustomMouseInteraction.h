#pragma once
#include "Module/MouseInputModule.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	class CustomMouseInteraction : public MouseInputModule
	{
	public:
		CustomMouseInteraction() {};
		virtual ~CustomMouseInteraction() {};

		DEF_INSTANCE_IN(TopologyModule, Topology, "");

	protected:
		void onEvent(PMouseEvent event) override;
	};
}
