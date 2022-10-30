#include "RigidBody.h"
#include "Quat.h"

namespace dyno
{
	IMPLEMENT_TCLASS(RigidBody, TDataType)

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