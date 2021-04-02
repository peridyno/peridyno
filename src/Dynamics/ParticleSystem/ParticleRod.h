#pragma once
#include "ParticleSystem.h"
#include <vector>

namespace dyno
{
	template<typename> class ElasticityModule;
	template<typename> class OneDimElasticityModule;
	template<typename> class ParticleIntegrator;
	template<typename> class FixedPoints;
	template<typename> class SimpleDamping;
	/*!
	*	\class	ParticleRod
	*	\brief	Peridynamics-based elastic object.
	*/
	template<typename TDataType>
	class ParticleRod : public ParticleSystem<TDataType>
	{
		DECLARE_CLASS_1(ParticleRod, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleRod(std::string name = "default");
		virtual ~ParticleRod();

		bool initialize() override;
		bool resetStatus() override;
		void advance(Real dt) override;

		void setParticles(std::vector<Coord> particles);

		void setLength(Real length);
		void setMaterialStiffness(Real stiffness);

		void addFixedParticle(int id, Coord pos);
		void removeFixedParticle(int id);

		void getHostPosition(std::vector<Coord>& pos);

		void removeAllFixedPositions();

		void doCollision(Coord pos, Coord dir);

		void setDamping(Real d);
	public:
		VarField<Real> m_horizon;

		VarField<Real> m_length;

		VarField<Real> m_stiffness;

	protected:
		DeviceArrayField<Real> m_mass;

	private:
		std::vector<int> m_fixedIds;

		void resetMassField();

		bool m_modifed = false;

		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
		std::shared_ptr<OneDimElasticityModule<TDataType>> m_one_dim_elasticity;
		std::shared_ptr<FixedPoints<TDataType>> m_fixed;
		std::shared_ptr<SimpleDamping<TDataType>> m_damping;
	};
}