#pragma once
#include "Module.h"

namespace dyno 
{
	class ComputeModule : public Module
	{
	public:
		ComputeModule();
		~ComputeModule() override;

		std::string getModuleType() override { return "ComputeModule"; }

	protected:
		virtual void compute() = 0;

	private:
		void updateImpl() final;
	};
}

