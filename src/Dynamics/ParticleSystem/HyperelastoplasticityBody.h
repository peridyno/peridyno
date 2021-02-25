#pragma once
#include "ParticleSystem/HyperelasticBody.h"

namespace dyno
{
	template<typename> class NeighborQuery;
	template<typename> class NeighborTetQuery;
	template<typename> class PointSetToPointSet;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;
	template<typename> class HyperelasticityModule_test;
	template<typename> class HyperelastoplasticityModule;
	template<typename> class HyperelasticFractureModule;
	template<typename> class DensityPBD;
	template<typename> class TetCollision;
	template<typename TDataType> class ImplicitViscosity;
	/*!
	*	\class	HyperelastoplasticityBody
	*	\brief	Peridynamics-based elastoplastic object.
	*/
	template<typename TDataType>
	class HyperelastoplasticityBody : public HyperelasticBody<TDataType>
	{
		DECLARE_CLASS_1(HyperelastoplasticityBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		HyperelastoplasticityBody(std::string name = "default");
		virtual ~HyperelastoplasticityBody();

		void advance(Real dt) override;

		void updateTopology() override;
		bool resetStatus() override;
		void updateStatus() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		//void setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver);

		std::shared_ptr<HyperelasticFractureModule<TDataType>> getFractureModule() {
			return m_fracture;
		}

		std::shared_ptr<Node> getSurfaceNode() { return m_surfaceNode; }

	protected:
		bool initialize() override;
		void updateRestShape() override;

		std::shared_ptr<Node> m_surfaceNode;

		DEF_VAR(CollisionEnabled, bool, true, "");

		DEF_EMPTY_CURRENT_ARRAY(PrincipleYielding, Coord, DeviceType::GPU, "Storing the plastic yielding along three principle stretches.");

		DEF_EMPTY_CURRENT_ARRAY(FractureTag, bool, DeviceType::GPU, "Indicating whether a triangle is to be separated");

		std::shared_ptr<ElasticityModule<TDataType>> m_linear_elasticity;
		std::shared_ptr<HyperelastoplasticityModule<TDataType>> m_plasticity;
		std::shared_ptr<HyperelasticFractureModule<TDataType>> m_fracture;
		std::shared_ptr<DensityPBD<TDataType>> m_pbdModule;
		std::shared_ptr<ImplicitViscosity<TDataType>> m_visModule;

		std::shared_ptr<HyperelasticityModule_test<TDataType>> m_hyper_new;

		std::shared_ptr<NeighborTetQuery<TDataType>> m_nbrTetQuery;
		std::shared_ptr<TetCollision<TDataType>> m_tetCollision;
	};

#ifdef PRECISION_FLOAT
	template class HyperelastoplasticityBody<DataType3f>;
#else
	template class HyperelastoplasticityBody<DataType3d>;
#endif
}