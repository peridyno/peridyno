

#pragma once
#include "Collision/CollisionData.h"
#include "Peridynamics/TetrahedralSystem.h"
#include "Module/ComputeModule.h"

namespace dyno 
{

	template<typename TDataType>
	class CalculateNormalSDF : public ComputeModule
	{
		DECLARE_TCLASS(CalculateNormalSDF, TDataType)
	public:
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Real Real;
		typedef typename TContactPair<Real> ContactPair;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;

		CalculateNormalSDF() {};
		~CalculateNormalSDF() override {};

		void compute() override;
		//void resetStates() override;

	public:

		DEF_ARRAY_IN(Coord, Position, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, NormalSDF, DeviceType::GPU, "");
		DEF_ARRAY_IN(Real, DisranceSDF, DeviceType::GPU, "");
		DEF_ARRAY_IN(Tetrahedron, Tets, DeviceType::GPU, "");
	};
}
