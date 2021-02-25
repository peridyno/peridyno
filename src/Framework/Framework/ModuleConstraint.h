#pragma once
#include "Framework/Module.h"

namespace dyno
{
class Field;

class ConstraintModule : public Module
{
public:
	ConstraintModule();
	~ConstraintModule() override;

	bool execute() override;

	virtual bool constrain() { return true; }

	std::string getModuleType() override { return "ConstraintModule"; }
};
}
