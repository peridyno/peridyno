#include "Sampler.h"

#include "GLPointVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	Sampler<TDataType>::Sampler()
		: Node()
	{
		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());

		auto glModule = std::make_shared<GLPointVisualModule>();
		glModule->varBaseColor()->setValue(Color(0.25f, 0.52f, 0.8f));
		glModule->varVisible()->setValue(true);
		glModule->varPointSize()->setValue(0.005);
		this->statePointSet()->connect(glModule->inPointSet());
		this->graphicsPipeline()->pushModule(glModule);
	}

	DEFINE_CLASS(Sampler);
}