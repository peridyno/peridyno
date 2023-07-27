#include "CubeSampler.h"

#include "GLPointVisualModule.h"
#include "SamplingPoints.h"
namespace dyno
{
	template<typename TDataType>
	SamplingPoints<TDataType>::SamplingPoints()
		: Node()
	{

		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());

		glModule = std::make_shared<GLPointVisualModule>();
		glModule->setColor(Color(0.25f, 0.52f, 0.8f));
		glModule->setVisible(true);
		glModule->varPointSize()->setValue(0.01);
		this->statePointSet()->connect(glModule->inPointSet());
		this->graphicsPipeline()->pushModule(glModule);
	}

	template<typename TDataType>
	void SamplingPoints<TDataType>::disableRender() {
		glModule->setVisible(false);
	};


	template<typename TDataType>
	int SamplingPoints<TDataType>::pointSize() {
		return this->statePointSet()->getData().getPointSize();
	};



	DEFINE_CLASS(SamplingPoints);
}