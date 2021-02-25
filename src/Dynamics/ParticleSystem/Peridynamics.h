#pragma once
#include <vector_types.h>
#include <vector>
#include "Framework/NumericalModel.h"
#include "ElasticityModule.h"
#include "Framework/FieldVar.h"


namespace dyno
{
	template<typename> class NeighborQuery;
	template<typename> class PointSetToPointSet;
	template<typename> class ParticleIntegrator;
	template<typename> class ElasticityModule;
	
	/*!
	*	\class	ParticleSystem
	*	\brief	Projective peridynamics
	*
	*	This class implements the projective peridynamics.
	*	Refer to He et al' "Projective peridynamics for modeling versatile elastoplastic materials" for details.
	*/
	template<typename TDataType>
	class Peridynamics : public NumericalModel
	{
		DECLARE_CLASS_1(Peridynamics, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Peridynamics();
		~Peridynamics() override {};

		/*!
		*	\brief	All variables should be set appropriately before initializeImpl() is called.
		*/
		bool initializeImpl() override;

		void step(Real dt) override;


	public:
		VarField<Real> m_horizon;

		DeviceArrayField<Coord> m_position;
		DeviceArrayField<Coord> m_velocity;
		DeviceArrayField<Coord> m_forceDensity;

	private:
		HostVarField<int>* m_num;
		HostVarField<Real>* m_mass;
		
		HostVarField<Real>* m_samplingDistance;
		HostVarField<Real>* m_restDensity;

		std::shared_ptr<PointSetToPointSet<TDataType>> m_mapping;
		std::shared_ptr<ParticleIntegrator<TDataType>> m_integrator;
		std::shared_ptr<NeighborQuery<TDataType>> m_nbrQuery;
		std::shared_ptr<ElasticityModule<TDataType>> m_elasticity;
	};

#ifdef PRECISION_FLOAT
	template class Peridynamics<DataType3f>;
#else
	template class Peridynamics<DataType3d>;
#endif
}