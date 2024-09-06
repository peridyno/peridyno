#pragma once
#include "ParticleSystem/ParticleSystem.h"

#include "Bond.h"

namespace dyno
{
	template<typename> class ElasticityModule;
	template<typename> class PointSetToPointSet;

	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ElasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_TCLASS(ElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TBond<TDataType> Bond;

		ElasticBody();
		~ElasticBody() override;

		void loadParticles(std::string filename);

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
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_ARRAY_STATE(Coord, ReferencePosition, DeviceType::GPU, "Reference position");

		DEF_ARRAYLIST_STATE(Bond, Bonds, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void updateTopology() override;
	};
}