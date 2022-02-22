#include "InputFieldTest.h"

namespace dyno
{
	template<typename TDataType>
	InputFieldTest<TDataType>::InputFieldTest()
		: Node()
	{
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
