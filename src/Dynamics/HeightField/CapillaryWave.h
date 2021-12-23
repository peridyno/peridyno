#pragma once
#include "Node.h"

namespace dyno
{
	/*!
	*	\class	CapillaryWave
	*	\brief	Peridynamics-based CapillaryWave.
	*/
	template<typename TDataType>
	class CapillaryWave : public Node
	{
		DECLARE_CLASS_1(CapillaryWave, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename Vector<Real, 2> Coord2D;
		typedef typename Vector<Real, 4> Coord4D;

		CapillaryWave(std::string name = "default");
		virtual ~CapillaryWave();

	public:

		DEF_ARRAY2D_STATE(Coord2D, Velocity, DeviceType::GPU, "Height field velocity");

	protected:
		void resetStates() override;

		void updateStates() override;

		void updateTopology() override;
	};
}