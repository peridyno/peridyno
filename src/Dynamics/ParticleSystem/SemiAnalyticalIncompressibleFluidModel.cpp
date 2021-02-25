#include "SemiAnalyticalIncompressibleFluidModel.h"
#include "Topology/PointSet.h"
#include "Framework/Node.h"
#include "ParticleIntegrator.h"
//#include "DensitySummation.h"
#include "ImplicitViscosity.h"
#include "SemiAnalyticalIncompressibilityModule.h"
#include "SurfaceTension.h"
#include "Helmholtz.h"
#include "Framework/MechanicalState.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/FieldNeighbor.h"
#include "Topology/NeighborQuery.h"
#include "Topology/NeighborTriangleQuery.h"
#include "ParticleSystem/Helmholtz.h"
#include "ParticleSystem/Attribute.h"
#include "Utility.h"
#include "Attribute.h"
#include "Framework/ModuleTopology.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(SemiAnalyticalIncompressibleFluidModel, TDataType)

	template<typename TDataType>
	SemiAnalyticalIncompressibleFluidModel<TDataType>::SemiAnalyticalIncompressibleFluidModel()
		: NumericalModel()
		, m_restRho(Real(1000))
		, m_pNum(0)
	{
		m_smoothing_length.setValue(Real(0.015));
		attachField(&m_smoothing_length, "smoothingLength", "Smoothing length", false);
		attachField(&m_particle_position, "position", "Storing the particle positions!", false);
		attachField(&m_particle_velocity, "velocity", "Storing the particle velocities!", false);
//		attachField(&m_force_density, "force_density", "Storing the particle force densities!", false);
	}

	template<typename TDataType>
	SemiAnalyticalIncompressibleFluidModel<TDataType>::~SemiAnalyticalIncompressibleFluidModel()
	{
		
	}

	template<typename TDataType>
	bool SemiAnalyticalIncompressibleFluidModel<TDataType>::initializeImpl()
	{
		printf("initialize fluid\n");

		m_nbrQueryPoint = this->getParent()->addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		m_smoothing_length.connect(m_nbrQueryPoint->inRadius());
		m_particle_position.connect(m_nbrQueryPoint->inPosition());
		
		m_nbrQueryPoint->initialize();

		


		m_integrator = this->getParent()->setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		m_particle_position.connect(m_integrator->inPosition());
		m_particle_velocity.connect(m_integrator->inVelocity());
		m_particle_force_density.connect(m_integrator->inForceDensity());
		m_integrator->initialize();
		


		m_nbrQueryTri = this->getParent()->addComputeModule<NeighborTriangleQuery<TDataType>>("neighborhood3");
		m_smoothing_length.connect(m_nbrQueryTri->inRadius());
		this->m_particle_position.connect(m_nbrQueryTri->inPosition());
		this->m_triangle_vertex.connect(m_nbrQueryTri->inTriangleVertex());
		this->m_triangle_index.connect(m_nbrQueryTri->inTriangleIndex());
		
		m_nbrQueryTri->initialize();
		
		
		m_visModule = this->getParent()->addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		m_smoothing_length.connect(&m_visModule->m_smoothingLength);
		this->m_particle_position.connect(&m_visModule->m_position);
		this->m_particle_velocity.connect(&m_visModule->m_velocity);
		m_nbrQueryPoint->outNeighborhood()->connect(&m_visModule->m_neighborhood);
		m_visModule->initialize();

		
		printf("finished initialize fluid viscosity\n");

		m_velocity_mod.setElementCount(m_particle_velocity.getElementCount());
		m_flip.setElementCount(m_particle_velocity.getElementCount());

		m_meshCollision = this->getParent()->addCollisionModel<MeshCollision<TDataType>>("mesh_collision");
		this->m_particle_position.connect(&m_meshCollision->m_position);
		this->m_particle_velocity.connect(&m_meshCollision->m_velocity);
		this->m_triangle_vertex.connect(&m_meshCollision->m_triangle_vertex);
		this->m_triangle_index.connect(&m_meshCollision->m_triangle_index);
		m_nbrQueryTri->outNeighborhood()->connect(&m_meshCollision->m_neighborhood_tri);
		m_meshCollision->initialize();
		this->m_velocity_mod.connect(&m_meshCollision->m_velocity_mod);
		m_flip.connect(&m_meshCollision->m_flip);
		//pReduce = Reduction<Real>::Create(m_velocity_mod.getElementCount());
		
		printf("finished initialize mesh collision\n");

		m_pbdModule = this->getParent()->addConstraintModule<SemiAnalyticalIncompressibilityModule<TDataType>>("velosity");
		m_smoothing_length.connect(&m_pbdModule->m_smoothing_length);

		m_particle_velocity.connect(&m_pbdModule->m_particle_velocity);

		m_particle_mass.connect(&m_pbdModule->m_particle_mass);
		this->m_particle_attribute.connect(&m_pbdModule->m_particle_attribute);

		
		this->m_particle_position.connect(&m_pbdModule->m_particle_position);

		this->m_triangle_vertex.connect(&m_pbdModule->m_triangle_vertex);
		this->m_triangle_vertex_old.connect(&m_pbdModule->m_triangle_vertex_old);

		this->m_triangle_vertex_mass.connect(&m_pbdModule->m_triangle_vertex_mass);

		this->m_triangle_index.connect(&m_pbdModule->m_triangle_index);

		m_nbrQueryPoint->outNeighborhood()->connect(&m_pbdModule->m_neighborhood_particles);
		m_flip.connect(&m_pbdModule->m_flip);
		m_nbrQueryTri->outNeighborhood()->connect(&m_pbdModule->m_neighborhood_triangles);
		m_pbdModule->initialize();

		

		return true;
	}

	template<typename TDataType>
	void SemiAnalyticalIncompressibleFluidModel<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}
		//return;
// 		if (first == 1)
// 		{
// 
// 			m_nbrQueryTri = this->getParent()->addComputeModule<NeighborQuery<TDataType>>("neighborhood3");
// 			m_smoothing_length.connect(m_nbrQueryTri->m_radius);
// 			this->m_particle_position.connect(m_nbrQueryTri->m_position);
// 			this->m_triangle_vertex.connect(m_nbrQueryTri->m_TriPos);
// 			this->m_triangle_index.connect(m_nbrQueryTri->m_triangls);
// 			m_nbrQueryTri->initialize();
// 			m_nbrQueryTri->m_neighborhood.connect(m_pbdModule->m_neighborhood_triangles);
// 
// 			m_pbdModule->initialize_();
// 			first = 0;
// 			return;
// 		
// 		}

		m_velocity_mod.setElementCount(m_particle_velocity.getElementCount());
		m_flip.setElementCount(m_particle_velocity.getElementCount());
		m_flip.getValue().reset();

		//m_integrator->begin();
		
		m_integrator->integrate();


		
		m_nbrQueryPoint->compute();

		
		m_nbrQueryTri->compute();



		//m_meshCollision->calculateVel();
		//Real max_vel;
		//max_vel = pReduce->maximum(m_velocity_mod.getValue()->getDataPtr(), m_velocity_mod.getElementCount());
		//var_smoothing_length.setValue(max(m_smoothing_length.getValue(), max_vel * dt));
		//m_nbrQueryTriMulti->compute();
		
		
		
		m_meshCollision->doCollision();
		
		
		m_visModule->constrain();
		
		
		m_pbdModule->constrain();
		
		m_integrator->end();
	}

	template<typename TDataType>
	void SemiAnalyticalIncompressibleFluidModel<TDataType>::setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_incompressibilitySolver)
		{
			getParent()->deleteConstraintModule(m_incompressibilitySolver);
		}
		m_incompressibilitySolver = solver;
		getParent()->addConstraintModule(solver);
	}


	template<typename TDataType>
	void SemiAnalyticalIncompressibleFluidModel<TDataType>::setViscositySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_viscositySolver)
		{
			getParent()->deleteConstraintModule(m_viscositySolver);
		}
		m_viscositySolver = solver;
		getParent()->addConstraintModule(solver);
	}



	template<typename TDataType>
	void SemiAnalyticalIncompressibleFluidModel<TDataType>::setSurfaceTensionSolver(std::shared_ptr<ConstraintModule> solver)
	{
		//if (!m_surfaceTensionSolver)
	//	{
	//		getParent()->deleteConstraintModule(m_surfaceTensionSolver);
	//	}
	//	m_surfaceTensionSolver = solver;
	//	getParent()->addConstraintModule(m_surfaceTensionSolver);
	}
}