#include "H_RigidBody.h"
#include "Quat.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(RigidBody, TDataType)

	template<typename TDataType>
	RigidBody<TDataType>::RigidBody(std::string name)
		: Node(name)
	{
	}

	template<typename TDataType>
	RigidBody<TDataType>::~RigidBody()
	{
	}

	DEFINE_CLASS(RigidBody);
}