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

		glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.3f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSet()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		this->inTriangleSet01()->tagOptional(true);
		this->inTriangleSet02()->tagOptional(true);
		this->inTriangleSet03()->tagOptional(true);
		this->inTriangleSet04()->tagOptional(true);

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
		auto tri01 = this->inTriangleSet01()->getDataPtr();
		auto tri02 = this->inTriangleSet02()->getDataPtr();
		auto tri03 = this->inTriangleSet03()->getDataPtr();
		auto tri04 = this->inTriangleSet04()->getDataPtr();

		std::shared_ptr<TriangleSet<TDataType>> temp = std::make_shared<TriangleSet<DataType3f>>();

		auto topo = this->stateTriangleSet()->getDataPtr();
		if (tri01 != NULL)
		{
			temp->copyFrom(*temp->merge(*tri01));
			printf("Merge TriangleSet01\n");
		}
		if (tri02 != NULL)
		{
			temp->copyFrom(*temp->merge(*tri02));
			printf("Merge TriangleSet02\n");
		}
		if (tri03 != NULL)
		{
			temp->copyFrom(*temp->merge(*tri03));
			printf("Merge TriangleSet03\n");
		}
		if (tri04 != NULL)
		{
			temp->copyFrom(*temp->merge(*tri04));
			printf("Merge TriangleSet04\n");
		}
		topo->copyFrom(*temp);
	}

	DEFINE_CLASS(Merge);
}