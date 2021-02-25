#include "PositionBasedFluidModelMesh.h"
#include "Topology/PointSet.h"
#include "Framework/Node.h"
#include "ParticleIntegrator.h"
#include "DensitySummationMesh.h"
#include "ImplicitViscosity.h"
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
#include "Topology/PointSet.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(PositionBasedFluidModelMesh, TDataType)

	template<typename TDataType>
	PositionBasedFluidModelMesh<TDataType>::PositionBasedFluidModelMesh()
		: NumericalModel()
		, m_restRho(Real(1000))
		, m_pNum(0)
	{
		m_smoothingLength.setValue(Real(0.015));
		attachField(&m_smoothingLength, "smoothingLength", "Smoothing length", false);
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		attachField(&m_forceDensity, "force_density", "Storing the particle force densities!", false);

		m_pSetGhost = std::make_shared<PointSet<TDataType>>();
	}

	template<typename TDataType>
	PositionBasedFluidModelMesh<TDataType>::~PositionBasedFluidModelMesh()
	{
		
	}

	

	template<typename TDataType>
	bool PositionBasedFluidModelMesh<TDataType>::initializeImpl()
	{

		//initGhostBoundary();
		//printf("INSIDE\n");
		Start.setValue(30000000);
		m_flip.setElementCount(m_position.getElementCount());

		m_nbrQueryPoint = this->getParent()->addComputeModule<NeighborQuery<TDataType>>("neighborhoodFluid");
		m_smoothingLength.connect(m_nbrQueryPoint->inRadius());
		m_position.connect(m_nbrQueryPoint->inPosition());
		m_nbrQueryPoint->initialize();




		m_nbrQueryTri = this->getParent()->addComputeModule<NeighborTriangleQuery<TDataType>>("neighborhoodTri");
		m_smoothingLength.connect(m_nbrQueryTri->inRadius());
		this->m_position.connect(m_nbrQueryTri->inPosition());
		this->TriPoint.connect(m_nbrQueryTri->inTriangleVertex());
		this->Tri.connect(m_nbrQueryTri->inTriangleIndex());
		m_nbrQueryTri->initialize();


		m_pbdModule2 = this->getParent()->addConstraintModule<DensityPBDMesh<TDataType>>("density_constraint");
		m_smoothingLength.connect(&m_pbdModule2->m_smoothingLength);
		m_position.connect(&m_pbdModule2->m_position);
		m_velocity.connect(&m_pbdModule2->m_velocity);
		m_nbrQueryPoint->outNeighborhood()->connect(&m_pbdModule2->m_neighborhood);
		m_nbrQueryTri->outNeighborhood()->connect(&m_pbdModule2->m_neighborhoodTri);
		Tri.connect(&m_pbdModule2->Tri);
		TriPoint.connect(&m_pbdModule2->TriPoint);
		Start.connect(&m_pbdModule2->Start);
		m_vn.connect(&m_pbdModule2->m_veln);

		//m_pbdModule2->initialize();
		//printf("###############&&&&&&&&&&&&&&&&&&&&****************\n");



		m_integrator = this->getParent()->setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		m_position.connect(m_integrator->inPosition());
		m_velocity.connect(m_integrator->inVelocity());
		m_forceDensity.connect(m_integrator->inForceDensity());
		m_integrator->initialize();

		m_visModule = this->getParent()->addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		m_smoothingLength.connect(&m_visModule->m_smoothingLength);
		m_position.connect(&m_visModule->m_position);
		m_velocity.connect(&m_visModule->m_velocity);
		m_nbrQueryPoint->outNeighborhood()->connect(&m_visModule->m_neighborhood);
		m_visModule->initialize();


		m_meshCollision = this->getParent()->addCollisionModel<MeshCollision<TDataType>>("mesh_collision");
		this->m_position.connect(&m_meshCollision->m_position);
		this->m_velocity.connect(&m_meshCollision->m_velocity);
		TriPoint.connect(&m_meshCollision->m_triangle_vertex);
		TriPointOld.connect(&m_meshCollision->m_triangle_vertex_old);
		Tri.connect(&m_meshCollision->m_triangle_index);
		m_nbrQueryTri->outNeighborhood()->connect(&m_meshCollision->m_neighborhood_tri);
		m_meshCollision->initialize();
		m_flip.connect(&m_meshCollision->m_flip);
		
		return true;
	}

	template<typename TDataType>
	void PositionBasedFluidModelMesh<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}

		

		m_integrator->begin();
		
		
		m_integrator->integrate();


		m_nbrQueryPoint->compute();
		m_nbrQueryTri->compute();
		
		
		

		printf("vis finished\n");
		m_meshCollision->doCollision();

		m_visModule->constrain();
		m_pbdModule2->constrain();
		//cudaMemcpy(m_position.getValue().getDataPtr(), m_position_all.getValue().getDataPtr(), m_position.getValue().size() * sizeof(Coord), cudaMemcpyDeviceToDevice);
		printf("pbd finished\n");
		//m_meshCollision->doCollision();

		
		m_integrator->end();
	}

	template<typename TDataType>
	void PositionBasedFluidModelMesh<TDataType>::setIncompressibilitySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_incompressibilitySolver)
		{
			getParent()->deleteConstraintModule(m_incompressibilitySolver);
		}
		m_incompressibilitySolver = solver;
		getParent()->addConstraintModule(solver);
	}


	template<typename TDataType>
	void PositionBasedFluidModelMesh<TDataType>::setViscositySolver(std::shared_ptr<ConstraintModule> solver)
	{
		if (!m_viscositySolver)
		{
			getParent()->deleteConstraintModule(m_viscositySolver);
		}
		m_viscositySolver = solver;
		getParent()->addConstraintModule(solver);
	}



	template<typename TDataType>
	void PositionBasedFluidModelMesh<TDataType>::setSurfaceTensionSolver(std::shared_ptr<ConstraintModule> solver)
	{
		//if (!m_surfaceTensionSolver)
	//	{
	//		getParent()->deleteConstraintModule(m_surfaceTensionSolver);
	//	}
	//	m_surfaceTensionSolver = solver;
	//	getParent()->addConstraintModule(m_surfaceTensionSolver);
	}
}