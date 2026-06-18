#include "VirtualParticleGenerator.h"

namespace dyno
{
	template<typename TDataType>
	VirtualParticleGenerator<TDataType>::VirtualParticleGenerator()
		: ConstraintModule()
	{
	}

	DEFINE_CLASS(VirtualParticleGenerator);
}