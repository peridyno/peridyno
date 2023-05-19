#pragma once
#include "Particlesystem/ParticleSystem.h"

#include "SharedDataInPeridynamics.h"

namespace dyno
{
	/*!
	*	\class	ParticleElastoplasticBody
	*	\brief	Peridynamics-based elastoplastic object.
	*/
	template<typename TDataType>
	class ElastoplasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(ElastoplasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		ElastoplasticBody();
		~ElastoplasticBody() override;

	public:
		FVar<Real> m_horizon;

		DEF_ARRAYLIST_STATE(NPair, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void updateTopology() override;
	};
}