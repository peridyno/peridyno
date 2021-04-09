#pragma once
#include "Framework/Module.h"

namespace dyno {

	class CustomModule : public Module
	{
		DECLARE_CLASS(CustomModule)
	public:
		CustomModule();
		virtual ~CustomModule();

		bool execute() override;

		std::string getModuleType() override { return "CustomModule"; }

	protected:
		virtual void applyCustomBehavior();

	};

}

