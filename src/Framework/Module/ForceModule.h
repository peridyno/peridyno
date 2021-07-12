#pragma once
#include "Module.h"

namespace dyno{

class ForceModule : public Module
{
	DECLARE_CLASS(ForceModule)
public:
	ForceModule();
	virtual ~ForceModule();

	virtual bool applyForce() { return true; }

	void setForceID(FieldID id) { m_forceID = id; }
	void setTorqueID(FieldID id) { m_torqueID = id; }

	std::string getModuleType() override { return "ForceModule"; }

protected:
	FieldID m_forceID;
	FieldID m_torqueID;
};
}

