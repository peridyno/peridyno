#include "CarDriver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CarDriver, TDataType);

	template<typename TDataType>
	CarDriver<TDataType>::CarDriver()
		: KeyboardInputModule()
	{


	}

	template<typename TDataType>
	void CarDriver<TDataType>::onEvent(PKeyboardEvent event)
	{
	
		Real epsilonAngle = M_PI / 360;
		auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->inTopology()->getDataPtr());

		auto& d_hinge = topo->hingeJoints();
		CArray<HingeJoint> c_hinge;
		c_hinge.assign(d_hinge);
		

		switch (event.key)
		{
		case PKeyboardType::PKEY_W:

			speed += 2.5;
			if (speed >= 30)
				speed = 30;
			for (size_t i = 0; i < 4; i++)
			{
				c_hinge[i].setMoter(speed);
			}
			d_hinge.assign(c_hinge);

			break;
		case PKeyboardType::PKEY_S:

			speed -= 2.5;
			if (speed <= -10)
				speed = -10;
			for (size_t i = 0; i < 4; i++)
			{
				c_hinge[i].setMoter(speed);
			}
			d_hinge.assign(c_hinge);

			break;
		case PKeyboardType::PKEY_A:
			for (size_t i = 4; i < 6; i++)
			{
				if(angle <= M_PI/12)
					angle += M_PI / 60;
				c_hinge[i].setRange(angle, angle + epsilonAngle);
			}
			d_hinge.assign(c_hinge);
			break;
		case PKeyboardType::PKEY_D:
			for (size_t i = 4; i < 6; i++)
			{
				if(angle >= -M_PI/12)
					angle -= M_PI / 60;
				c_hinge[i].setRange(angle, angle + epsilonAngle);
			}
			d_hinge.assign(c_hinge);
			break;
		case PKeyboardType::PKEY_Q:
			for (size_t i = 4; i < 6; i++)
			{
				angle = 0.0f;
				c_hinge[i].setRange(angle, angle + epsilonAngle);
			}
			d_hinge.assign(c_hinge);
			break;

		case PKeyboardType::PKEY_E:
			speed = 0;
			for (size_t i = 0; i < 4; i++)
			{
				c_hinge[i].setMoter(speed);
			}
			d_hinge.assign(c_hinge);
			break;
		default:
			break;
		}

	}
	
	DEFINE_CLASS(CarDriver);
}