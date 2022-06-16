#pragma once
#include "Module/InputMouseModule.h"
#include "Module/TopologyModule.h"

namespace dyno
{
	class CustomMouseIteraction : public InputMouseModule
	{
	public:
		CustomMouseIteraction() {};
		virtual ~CustomMouseIteraction() {};

		DEF_INSTANCE_IN(TopologyModule, Topology, "");

	protected:
		void onEvent(PMouseEvent event) override;
	};
}
