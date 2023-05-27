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
		virtual ~ElasticBody();

		void updateTopology() override;

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_ARRAY_STATE(Coord, ReferencePosition, DeviceType::GPU, "Reference position");

		DEF_ARRAYLIST_STATE(Bond, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

// 	private:
// 		std::shared_ptr<Node> m_surfaceNode;
	};
}