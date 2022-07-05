#pragma once
#include "Module/InputMouseModule.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	class CustomMouseInteraction : public InputMouseModule
	{
	public:
		CustomMouseInteraction() {};
		virtual ~CustomMouseInteraction() {};

		DEF_INSTANCE_IN(TopologyModule, Topology, "");

	protected:
		void onEvent(PMouseEvent event) override;
	};
}
