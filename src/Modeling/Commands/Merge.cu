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
		this->stateTriangleSets()->setDataPtr(std::make_shared<TriangleSets<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.3f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSets()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto ptModule = std::make_shared<GLPointVisualModule>();
		ptModule->setVisible(false);
		this->stateTriangleSets()->connect(ptModule->inPointSet());
		this->graphicsPipeline()->pushModule(ptModule);
		ptModule->varPointSize()->setValue(0.01);

		this->stateTriangleSets()->promoteOuput();
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
		auto num = this->inTriangleSets()->size();

		std::vector<std::shared_ptr<TriangleSet<TDataType>>> tsArray;
		for (uint i = 0; i < num; i++)
		{
			auto ts = this->inTriangleSets()->constDataPtr(i);

			tsArray.push_back(ts);
		}

		auto topo = this->stateTriangleSets()->getDataPtr();
		topo->load(tsArray);

		tsArray.clear();
	}

	DEFINE_CLASS(Merge);
}