#pragma once
#include "Framework/ModuleForce.h"

namespace dyno{

template<typename TDataType>
class Gravity : public ForceModule
{
	DECLARE_CLASS_1(Gravity, TDataType)
public:
	typedef typename TDataType::Real Real;
	typedef typename TDataType::Coord Coord;

	Gravity();
	virtual ~Gravity();

	bool applyForce() override;

	void setGravity(Coord g) { m_gravity = g; }

	void setMassID(FieldID id) { m_massID = id; }
protected:
	FieldID m_massID;

private:
	Coord m_gravity;
};

#ifdef PRECISION_FLOAT
template class Gravity<DataType3f>;
#else
template class Gravity<DataType3d>;
#endif

}

