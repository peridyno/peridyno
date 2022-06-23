#include "SphereModel.h"

namespace dyno
{
	template<typename TDataType>
	SphereModel<TDataType>::SphereModel()
		: ParametricModel<TDataType>()
	{
		
	}

	template<typename TDataType>
	void SphereModel<TDataType>::resetStates()
	{

	}

	DEFINE_CLASS(SphereModel);
}