#pragma once
#include "Module.h"

namespace dyno
{

class IOModule : public Module
{
public:
	IOModule();
	virtual ~IOModule();

	virtual void display() {};

	void enable(bool bEnable) { m_enabled = bEnable; }
	bool isEnabled() { return m_enabled; }

	std::string getModuleType() override { return "IOModule"; }
protected:
	bool m_enabled;
};

}
