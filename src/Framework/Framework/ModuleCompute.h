#pragma once
#include "Framework/Module.h"

namespace dyno{

class ComputeModule : public Module
{
public:
	ComputeModule();
	~ComputeModule() override;

	bool execute() override;

	virtual void compute() {};

	std::string getModuleType() override { return "ComputeModule"; }
private:

};
}

