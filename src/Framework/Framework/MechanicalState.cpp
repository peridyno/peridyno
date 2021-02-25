#include "MechanicalState.h"

namespace dyno
{
	IMPLEMENT_CLASS(MechanicalState)

	MechanicalState::MechanicalState()
	{

	}

	MechanicalState::~MechanicalState(void)
	{

	}

	Real MechanicalState::getTotalMass()
	{
		return m_totalMass;
	}

	void MechanicalState::setTotalMass(Real mass)
	{
		m_totalMass = mass;
	}

	int MechanicalState::getDOF()
	{
		return 0;
	}

// 	void MechanicalState::resetForce()
// 	{
// 		resetField(MechanicalState::force());
// 		resetField(MechanicalState::torque());
// 	}

// 	void MechanicalState::resetField(std::string name)
// 	{
// 		auto field = this->getField(name);
// 		if (field != nullptr)
// 		{
// 			field->reset();
// 		}
// 	}

	FieldID MechanicalState::addAuxiliaryID(FieldID id)
	{
		m_auxIDs.insert(id);
		return id;
	}

	void MechanicalState::deleteAuxiliaryID(FieldID id)
	{
		if (hasAuxiliaryID(id))
		{
			m_auxIDs.erase(id);
		}
	}

	void MechanicalState::clearAllIDs()
	{
		m_auxIDs.clear();
	}

	bool MechanicalState::hasAuxiliaryID(FieldID id)
	{
		auto ret = m_auxIDs.find(id);
		if (ret == m_auxIDs.end())
		{
			return false;
		}

		return true;
	}

}