#pragma once
#include "TriangularSystem.h"
#include "SharedDataInPeridynamics.h"

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
		typedef TPair<TDataType> NPair;

		Cloth();
		~Cloth() override;

	public:
		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "Input");

		DEF_ARRAY_STATE(Coord, OldPosition, DeviceType::GPU, "");

		DEF_ARRAYLIST_STATE(NPair, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

	private:
		
	};
}