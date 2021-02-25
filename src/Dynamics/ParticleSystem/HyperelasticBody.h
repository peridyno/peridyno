#pragma once
#include "ParticleSystem/ParticleSystem.h"
#include "Topology/TetrahedronSet.h"
#include "NeighborData.h"
#include "TetSystem.h"
#include "Attribute.h"

namespace dyno
{
	template<typename> class ElasticityModule;
	template<typename> class ParticleIntegrator;
	template<typename> class TetrahedronSet;
	template<typename> class NeighborQuery;
	template<typename> class HyperelasticityModule_test;


	template<typename ElementType> using DynamicArray = std::vector<std::set<ElementType>>;

	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class HyperelasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename TPair<TDataType> NPair;

		HyperelasticBody(std::string name = "default");
		virtual ~HyperelasticBody();

		bool initialize() override;
		bool resetStatus() override;

		void advance(Real dt) override;
		void updateTopology() override;

		virtual bool translate(Coord t);
		virtual bool scale(Real s);

		void setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);

		std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();

		void loadCentroidsFromFile(std::string filename);
		void loadVertexFromFile(std::string filename);

		void loadStandardTet();
		void loadStandardSimplex();

		std::shared_ptr<TetSystem<TDataType>> getMeshNode() { return m_mesh_node; }

		void loadParticles(Coord lo, Coord hi, Real distance);

	public:
		DEF_EMPTY_VAR(Horizon, Real, "Horizon");

		DEF_EMPTY_CURRENT_ARRAY(RestPosition, Coord, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_ARRAY(VertexRotation, Matrix, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_ARRAY(Attribute, Attribute, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_ARRAY(Volume, Real, DeviceType::GPU, "");

		DEF_EMPTY_CURRENT_NEIGHBOR_LIST(RestShape, NPair, "");

	protected:
		void updateRestShape(std::vector<Coord>& points, std::vector<TopologyModule::Tetrahedron>& tets);

		virtual void updateRestShape();
		virtual void updateVolume();

		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;

		std::shared_ptr<ElasticityModule<TDataType>> m_linearElasticity;
		std::shared_ptr<HyperelasticityModule_test<TDataType>> m_hyper;

		std::shared_ptr<TetrahedronSet<TDataType>> m_pSet;

		std::shared_ptr<TetSystem<TDataType>> m_mesh_node;
	};


#ifdef PRECISION_FLOAT
	template class HyperelasticBody<DataType3f>;
#else
	template class HyperelasticBody<DataType3d>;
#endif
}