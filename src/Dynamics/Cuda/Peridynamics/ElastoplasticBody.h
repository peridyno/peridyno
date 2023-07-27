#pragma once
#include "Particlesystem/ParticleSystem.h"

#include "Bond.h"

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
		typedef typename TBond<TDataType> Bond;

		ElastoplasticBody();
		~ElastoplasticBody() override;

		void loadParticles(Coord lo, Coord hi, Real distance);

		bool translate(Coord t) {
			auto ptSet = this->statePointSet()->getDataPtr();
			ptSet->translate(t);

			return true;
		}

		bool scale(Real s) {
			auto ptSet = this->statePointSet()->getDataPtr();
			ptSet->scale(s);

			return true;
		}

		bool rotate(Quat<Real> q) {
			auto ptSet = this->statePointSet()->getDataPtr();
			ptSet->rotate(q);

			return true;
		}

	public:
		FVar<Real> m_horizon;

		DEF_ARRAYLIST_STATE(Bond, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void updateTopology() override;
	};
}