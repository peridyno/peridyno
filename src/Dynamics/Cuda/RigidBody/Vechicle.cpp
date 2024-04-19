#include "Vechicle.h"

namespace dyno
{
	template<typename TDataType>
	Vechicle<TDataType>::Vechicle()
		: RigidBodySystem<TDataType>()
	{
	}

	template<typename TDataType>
	Vechicle<TDataType>::~Vechicle()
	{

	}

	template<typename TDataType>
	void Vechicle<TDataType>::resetStates()
	{
		auto texMesh = this->inTextureMesh()->constDataPtr();

		uint N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		tms.resize(N, 1);

		for (uint i = 0; i < N; i++)
		{
			tms[i].insert(texMesh->shapes()[i]->boundingTransform);
		}

		if (this->stateInstanceTransform()->isEmpty())
		{
			this->stateInstanceTransform()->allocate();
		}

		auto instantanceTransform = this->stateInstanceTransform()->getDataPtr();
		instantanceTransform->assign(tms);

		tms.clear();
	}

	float dx = 0.1f;
	float theta = 0;
	template<typename TDataType>
	void Vechicle<TDataType>::updateStates()
	{
		auto texMesh = this->inTextureMesh()->constDataPtr();

		uint N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		tms.resize(N, 1);

		dx += 0.01f;
		theta += 0.01f;

		for (uint i = 0; i < N; i++)
		{
			Transform3f t =  texMesh->shapes()[i]->boundingTransform;
			//t.translation() = Vec3f(0, 0, 0);
			//t.rotation() = Quat1f(theta, Vec3f(1, 0, 0)).toMatrix3x3();
			tms[i].insert(t);
		}

		auto instantanceTransform = this->stateInstanceTransform()->getDataPtr();
		instantanceTransform->assign(tms);

		tms.clear();

		RigidBodySystem<TDataType>::updateStates();
	}

	DEFINE_CLASS(Vechicle);
}