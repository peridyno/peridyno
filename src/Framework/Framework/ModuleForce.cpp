#include "ModuleForce.h"
#include "Framework/Node.h"

namespace dyno
{
IMPLEMENT_CLASS(ForceModule)

ForceModule::ForceModule()
	: Module()
	, m_forceID(MechanicalState::force())
	, m_torqueID(MechanicalState::torque())
{
}

ForceModule::~ForceModule()
{
}

}