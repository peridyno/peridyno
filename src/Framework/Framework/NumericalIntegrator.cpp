#include "NumericalIntegrator.h"
#include "MechanicalState.h"

namespace dyno
{
	NumericalIntegrator::NumericalIntegrator()
		: Module()
		, m_massID(MechanicalState::mass())
		, m_forceID(MechanicalState::force())
		, m_torqueID(MechanicalState::torque())
		, m_posID(MechanicalState::position())
		, m_velID(MechanicalState::velocity())
		, m_posPreID(MechanicalState::pre_position())
		, m_velPreID(MechanicalState::pre_velocity())
	{

	}

	NumericalIntegrator::~NumericalIntegrator()
	{

	}
}