#pragma once
#include "Module.h"

namespace dyno {

	class CustomModule : public Module
	{
		DECLARE_CLASS(CustomModule)
	public:
		CustomModule();
		virtual ~CustomModule();

		std::string getModuleType() override { return "CustomModule"; }

	protected:
		void updateImpl() override;

		virtual void applyCustomBehavior();
	};

}

