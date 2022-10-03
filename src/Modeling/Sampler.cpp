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
		glModule->setColor(Vec3f(0.25, 0.52, 0.8));
		glModule->setVisible(true);
		glModule->varPointSize()->setValue(0.005);
		this->statePointSet()->connect(glModule->inPointSet());
		this->graphicsPipeline()->pushModule(glModule);
	}

	DEFINE_CLASS(Sampler);
}