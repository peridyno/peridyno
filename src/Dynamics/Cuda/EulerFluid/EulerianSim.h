#pragma once
#include "Node.h"
//#include "Topology/AdaptiveGridSet.h"
//#include "Topology/TriangleSet.h"

namespace dyno
{
	enum CellType
	{
		Undefined = 0,
		Inside,
		Inlet1,
		Inlet2,
		Outlet1,
		Outlet2,
		Static
	};

	template<typename TDataType>
	class EulerianSim : public Node
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		EulerianSim();
		~EulerianSim() override;

		//DEF_INSTANCE_STATE(AdaptiveGridSet<TDataType>, AdaptiveVolume, "The adaptive volume data of model");

		DEF_ARRAY_OUT(Coord, Velocity, DeviceType::GPU, "The velocity of Leafs");
		DEF_ARRAY_OUT(Real, Pressure, DeviceType::GPU, "The pressure of Leafs");

		Real total_time = 0.0f;
		Real m_density = 1060.0f;
		Real kinetic_coefficient = 0.004f;
		Real gravity = 9.8f;

	private:

	};
}