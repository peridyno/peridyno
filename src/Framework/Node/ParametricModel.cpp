#include "ParametricModel.h"

namespace dyno
{
	template<typename TDataType>
	ParametricModel<TDataType>::ParametricModel()
		: Node()
	{
		this->setForceUpdate(false);
		this->setAutoHidden(true);

		this->varScale()->setRange(Real(0.0001), Real(1000));
	}

	DEFINE_CLASS(ParametricModel);
}