#pragma once
#include "RigidBodySystem.h"

#include "Topology/TextureMesh.h"

#include "STL/Pair.h"

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
		typedef typename Pair<uint, uint> BindingPair;

		Vechicle();
		~Vechicle() override;

		void bind(uint bodyId, Pair<uint, uint> shapeId);

		void setInitialCenterDiff(std::vector<Vec3f> diff);

	public:

		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "Texture mesh of the vechicle");

		DEF_ARRAY_STATE(BindingPair, Binding, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

	protected:
		void resetStates() override;

		void updateStates() override;


	private:

		std::vector<Pair<uint, uint>> mBindingPair;

		DArray<Matrix> mInitialRot;

		DArray<Vec3f> mDiff;


	};

	IMPLEMENT_TCLASS(Vechicle, TDataType)
}
