#pragma once
#include "RigidBodySystem.h"

#include "Topology/TextureMesh.h"

namespace dyno 
{
	template<typename TDataType>
	class Vechicle : virtual public RigidBodySystem<TDataType>
	{
		DECLARE_TCLASS(Vechicle, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		Vechicle();
		~Vechicle() override;

	public:

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "Texture mesh of the vechicle");

		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

	protected:
		void resetStates() override;

		void updateStates() override;
	};

	IMPLEMENT_TCLASS(Vechicle, TDataType)
}
