#pragma once
#include "Module/MouseInputModule.h"
#include "Topology.h"

namespace dyno
{
	class CustomMouseInteraction : public MouseInputModule
	{
	public:
		CustomMouseInteraction() {};
		virtual ~CustomMouseInteraction() {};

		DEF_INSTANCE_IN(Topology, Topology, "");

	protected:
		void onEvent(PMouseEvent event) override;
	};
}
