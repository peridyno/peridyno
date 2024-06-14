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
	
		
		//auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->inTopology()->getDataPtr());

		//auto& d_hinge = topo->hingeJoints();
		//CArray<HingeJoint> c_hinge;
		//c_hinge.assign(d_hinge);
		//for (size_t i = 0; i < c_hinge.size(); i++)
		//{
		//	c_hinge[i].setMoter(1);
		//	c_hinge[i].setAxis(Vec3f(1, 0, 0));
		//}
		//d_hinge.assign(c_hinge);

		//auto& q = this->inQuaternion()->getData();

		switch (event.key)
		{
		case PKeyboardType::PKEY_W:

			//for (size_t i = 0; i < c_hinge.size(); i++)
			//{
			//	c_hinge[i].setMoter(1);
			//	c_hinge[i].setAxis(Vec3f(1, 0, 0));
			//}
			//d_hinge.assign(c_hinge);
			//printf("Key:W\n");
			//printf("size:%d\n",c_hinge.size());
			break;
		case PKeyboardType::PKEY_S:

			//for (size_t i = 0; i < c_hinge.size(); i++)
			//{
			//	c_hinge[i].setMoter(-10);
			//	c_hinge[i].setAxis(Vec3f(1, 0, 0));
			//}

			break;
		case PKeyboardType::PKEY_A:

			break;
		case PKeyboardType::PKEY_D:

			break;
		default:
			break;
		}

	}
	
	DEFINE_CLASS(CarDriver);
}