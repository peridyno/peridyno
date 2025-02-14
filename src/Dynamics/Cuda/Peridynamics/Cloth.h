#pragma once
#include "TriangularSystem.h"
#include "Bond.h"

namespace dyno
{
	/*!
	*	\class	Cloth
	*	\brief	Peridynamics-based cloth.
	*/
	template<typename TDataType>
	class Cloth : public TriangularSystem<TDataType>
	{
		DECLARE_TCLASS(Cloth, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::TBond<TDataType> Bond;

		Cloth();
		~Cloth() override;

		DEF_VAR(Real, Horizon, 0.01, "Horizon");

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "Input");

		DEF_VAR_STATE(Real, Horizon, Real(1), "A state field representing horizon");

		DEF_ARRAY_STATE(Coord, RestPosition, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Coord, OldPosition, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(Bond, Bonds, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

	private:
		
	};
}