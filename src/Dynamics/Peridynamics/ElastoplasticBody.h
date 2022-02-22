#pragma once
#include "Particlesystem/ParticleSystem.h"
#include "NeighborData.h"

namespace dyno
{
	template<typename> class NeighborPointQuery;
	template<typename> class PointSetToPointSet;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;
	template<typename> class ElastoplasticityModule;
	template<typename> class DensityPBD;
	template<typename TDataType> class ImplicitViscosity;
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

		ElastoplasticBody(std::string name = "default");
		virtual ~ElastoplasticBody();

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void loadSurface(std::string filename);

		std::shared_ptr<Node> getSurfaceNode() { return m_surfaceNode; }

	public:
		FVar<Real> m_horizon;

		DEF_EMPTY_CURRENT_ARRAYLIST(NPair, RestShape, DeviceType::GPU, "Storing neighbors");

	protected:
		void resetStates() override;

		void updateTopology() override;

	private:
		std::shared_ptr<Node> m_surfaceNode;

		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborPointQuery<TDataType>> m_nbrQuery;
		std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
		std::shared_ptr<ElastoplasticityModule<TDataType>> m_plasticity;
		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;
	};
}