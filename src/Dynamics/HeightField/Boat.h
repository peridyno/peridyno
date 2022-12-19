#pragma once
#include "Node.h"
#include "RigidBody/RigidBodySystem.h"
#include "Topology/TriangleSet.h"
#include "FilePath.h"
namespace dyno {

	template<typename TDataType>
	class Boat : public RigidBodySystem<TDataType>
	{
		DECLARE_TCLASS(Boat, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Boat();
		~Boat();

	protected:
		void resetStates() override;

	public:
		DEF_VAR(FilePath, FileName, "", "");

	};
	IMPLEMENT_TCLASS(Boat, TDataType)
}
