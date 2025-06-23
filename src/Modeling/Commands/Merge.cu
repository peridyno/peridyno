#include "Merge.h"
#include "Topology/PointSet.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"



namespace dyno
{
	template<typename TDataType>
	Merge<TDataType>::Merge()
	{
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.3f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto ptModule = std::make_shared<GLPointVisualModule>();
		ptModule->setVisible(false);
		this->stateTriangleSet()->connect(ptModule->inPointSet());
		this->graphicsPipeline()->pushModule(ptModule);
		ptModule->varPointSize()->setValue(0.01);

		this->stateTriangleSet()->promoteOuput();
	}

	template<typename TDataType>
	void Merge<TDataType>::resetStates()
	{
		MergeGPU();
	}
	template<typename TDataType>
	void Merge<TDataType>::preUpdateStates()
	{
		Node::preUpdateStates();
		if (this->varUpdateMode()->getData() == UpdateMode::Tick)
		{
			MergeGPU();
		}
	}

	template<typename TDataType>
	void Merge<TDataType>::MergeGPU()
	{
		std::shared_ptr<TriangleSet<TDataType>> temp = std::make_shared<TriangleSet<DataType3f>>();

		auto num = this->inTriangleSets()->size();

		for (uint i = 0; i < num; i++)
		{
			auto ts = this->inTriangleSets()->constDataPtr(i);

			temp->copyFrom(*temp->merge(*ts));
		}

		auto topo = this->stateTriangleSet()->getDataPtr();
		topo->copyFrom(*temp);
	}

	DEFINE_CLASS(Merge);
}