#pragma once
#include "ThreadSystem.h"
#include "Bond.h"

namespace dyno
{
	/*!
	*	\class	Thread
	*	\brief	Peridynamics-based Thread.
	*/
	template<typename TDataType>
	class Thread : public ThreadSystem<TDataType>
	{
		DECLARE_TCLASS(Thread, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TBond<TDataType> Bond;

		Thread();
		virtual ~Thread();

		bool translate(Coord t);
		bool scale(Real s);


	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_ARRAYLIST_STATE(Bond, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		

	private:
		
	};
}