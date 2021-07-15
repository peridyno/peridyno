#pragma once
#include "Module.h"

namespace dyno 
{
	class ComputeModule : public Module
	{
	public:
		ComputeModule();
		~ComputeModule() override;

		virtual void compute() = 0;

		std::string getModuleType() override { return "ComputeModule"; }
	private:
		void updateImpl() final;
	};
}

