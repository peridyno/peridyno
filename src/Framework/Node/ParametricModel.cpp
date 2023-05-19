#include "ParametricModel.h"

namespace dyno
{
	template<typename TDataType>
	ParametricModel<TDataType>::ParametricModel()
		: Node()
	{
		this->varScale()->setRange(Real(0.0001), Real(1000));

		this->allowExported(false);
	}

	DEFINE_CLASS(ParametricModel);
}