#pragma once
#include "ParticleSystem/ParticleSystem.h"
#include "NeighborData.h"

namespace dyno
{
	template<typename> class ElasticityModule;
	template<typename> class PointSetToPointSet;

	/*!
	*	\class	ParticleElasticBody
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ParticleElasticBody : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleElasticBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef TPair<TDataType> NPair;

		ParticleElasticBody(std::string name = "default");
		virtual ~ParticleElasticBody();

		bool initialize() override;
		void advance(Real dt) override;
		void updateTopology() override;
		bool resetStatus() override;

		bool translate(Coord t) override;
		bool scale(Real s) override;

		void setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver);
		std::shared_ptr<ElasticityModule<TDataType>> getElasticitySolver();
		void loadSurface(std::string filename);

		std::shared_ptr<PointSetToPointSet<TDataType>> getTopologyMapping();

		std::shared_ptr<Node> getSurfaceNode() { return m_surfaceNode; }

	public:
		DEF_VAR(Real, Horizon, 0.01, "Horizon");

		DEF_EMPTY_CURRENT_ARRAY(ReferencePosition, Coord, DeviceType::GPU, "Reference position");

		DEF_EMPTY_CURRENT_ARRAYLIST(int, NeighborIds, DeviceType::GPU, "Storing the ids for neighboring particles");

		DEF_EMPTY_CURRENT_ARRAYLIST(NPair, RestShape, DeviceType::GPU, "Storing neighbors");

	private:
		std::shared_ptr<Node> m_surfaceNode;
	};
}