#pragma once
#include "Module.h"

namespace dyno
{
	class FBase;

	class ConstraintModule : public Module
	{
	public:
		ConstraintModule();
		~ConstraintModule() override;

		virtual void constrain() = 0;

		std::string getModuleType() override { return "ConstraintModule"; }

	protected:
		void updateImpl() override;
	};
}
