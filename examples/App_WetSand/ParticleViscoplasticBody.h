#pragma once
#include "ParticleSystem/ParticleSystem.h"

namespace dyno
{
	template<typename> class NeighborQuery;
	template<typename> class PointSetToPointSet;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;
	template<typename> class ElastoplasticityModule;
	template<typename> class DensityPBD;
	template<typename TDataType> class ImplicitViscosity;
	/*!
	*	\class	ParticleViscoplasticBody
	*	\brief	Peridynamics-based elastoplastic object.
	*/
	template<typename TDataType>
	class ParticleViscoplasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleViscoplasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleViscoplasticBody(std::string name = "default");
		virtual ~ParticleViscoplasticBody();

		void advance(Real dt) override;

		void updateTopology() override;

		bool initialize() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void loadSurface(std::string filename);

	public:
		VarField<Real> m_horizon;

	private:
		std::shared_ptr<Node> m_surfaceNode;

		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
		std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
		std::shared_ptr<ElastoplasticityModule<TDataType>> m_plasticity;
		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;
	};


#ifdef PRECISION_FLOAT
	template class ParticleViscoplasticBody<DataType3f>;
#else
	template class ParticleViscoplasticBody<DataType3d>;
#endif
}