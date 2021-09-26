#pragma once
#include "Module.h"

namespace dyno
{
	class TopologyMapping : public Module
	{
	public:
		TopologyMapping();
		virtual ~TopologyMapping();

		virtual bool apply() = 0;

	private:
		void updateImpl() override;
	};
}