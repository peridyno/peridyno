#include "InputFieldTest.h"

namespace dyno
{
	template<typename TDataType>
	InputFieldTest<TDataType>::InputFieldTest()
		: Node()
	{
		this->varVariable()->setMax(Real(20.0f));
		this->varVariable()->setMin(Real(-10.0f));
	}

	template<typename TDataType>
	void InputFieldTest<TDataType>::updateStates()
	{
		if (!this->inPointSet()->isEmpty()) {
			printf("PointSet in InputFieldTest was set \n");
		}
	}

	DEFINE_CLASS(InputFieldTest);
}
