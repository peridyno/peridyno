#pragma once
#include "Framework/Module.h"

namespace dyno
{
	class FieldBase;

	class ConstraintModule : public Module
	{
	public:
		ConstraintModule();
		~ConstraintModule() override;

		virtual bool constrain() { return true; }

		std::string getModuleType() override { return "ConstraintModule"; }

	protected:
		bool updateImpl() override;
	};
}
