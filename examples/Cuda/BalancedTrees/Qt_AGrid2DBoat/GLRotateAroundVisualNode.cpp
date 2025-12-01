#include "GLRotateAroundVisualNode.h"
#include "Module/GLPhotorealisticInstanceRender.h"

namespace dyno
{
	IMPLEMENT_TCLASS(GLRotateAroundVisualNode, TDataType)

	template<typename TDataType>
	GLRotateAroundVisualNode<TDataType>::GLRotateAroundVisualNode()
		: Node()
	{
		auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
		this->stateTextureMesh()->connect(prRender->inTextureMesh());
		this->stateInstanceTransform()->connect(prRender->inTransform());
		this->graphicsPipeline()->pushModule(prRender);
	}

	template<typename TDataType>
	GLRotateAroundVisualNode<TDataType>::~GLRotateAroundVisualNode()
	{
	}

	template<typename TDataType>
	void GLRotateAroundVisualNode<TDataType>::resetStates()
	{
		auto shapes = this->getShapes();
		CArrayList<Transform3f> tms;
		tms.resize(1.0, shapes.size());
		for (int i = 0; i < shapes.size(); i++)
		{
			Quat<Real> q = shapes[i]->computeQuaternion();
			q.normalize();
			Coord3D tran(shapes[i]->varRotationRadius()->getData(), 0.0f, 0.0f);

			tms[0].insert(Transform3f(q.rotate(tran), q.toMatrix3x3()));
		}

		this->stateInstanceTransform()->assign(tms);
	}

	template<typename TDataType>
	void GLRotateAroundVisualNode<TDataType>::updateStates()
	{
		this->resetStates();
	}

	template<typename TDataType>
	bool GLRotateAroundVisualNode<TDataType>::validateInputs()
	{
		auto shapes = this->getShapes();
		if (shapes.size() == 0) return false;
	}

	DEFINE_CLASS(GLRotateAroundVisualNode);
}
