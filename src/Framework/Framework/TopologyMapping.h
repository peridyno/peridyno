#pragma once
#include "Framework/Module.h"

namespace dyno
{
	class TopologyMapping : public Module
	{
	public:
		TopologyMapping();
		virtual ~TopologyMapping();

		bool execute() override;

		virtual bool apply() = 0;
	private:

	};
}