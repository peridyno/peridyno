/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-08-06
 * @description: implemendation of PositionBasedFluidModelMesh class, a container for semi-analytical PBD fluids 
 *               introduced in the paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.1
 */
#include "SemiAnalyticalPositionBasedFluidModel.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "ParticleSystem/Module/IterativeDensitySolver.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"

namespace dyno {
	IMPLEMENT_TCLASS(SemiAnalyticalPositionBasedFluidModel, TDataType)

		template <typename TDataType>
	SemiAnalyticalPositionBasedFluidModel<TDataType>::SemiAnalyticalPositionBasedFluidModel()
		: GroupModule()
	{
		this->varSmoothingLength()->setValue(Real(0.0125));

		auto m_nbrQueryPoint = std::make_shared<NeighborPointQuery<TDataType>>();
		this->varSmoothingLength()->connect(m_nbrQueryPoint->inRadius());
		this->inPosition()->connect(m_nbrQueryPoint->inPosition());
		this->pushModule(m_nbrQueryPoint);
		//m_nbrQueryPoint->initialize();

		auto m_nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		this->varSmoothingLength()->connect(m_nbrQueryTri->inRadius());
		this->inPosition()->connect(m_nbrQueryTri->inPosition());
		this->inTriangleVertex()->connect(m_nbrQueryTri->inTriPosition());
		this->inTriangleIndex()->connect(m_nbrQueryTri->inTriangles());
		this->pushModule(m_nbrQueryTri);
		
		auto m_pbdModule2 = std::make_shared<SemiAnalyticalPBD<TDataType>>();
		this->inTimeStep()->connect(m_pbdModule2->inTimeStep());
		this->inPosition()->connect(m_pbdModule2->inPosition());
		this->inVelocity()->connect(m_pbdModule2->inVelocity());
		this->varSmoothingLength()->connect(m_pbdModule2->inSmoothingLength());
		m_nbrQueryPoint->outNeighborIds()->connect(m_pbdModule2->inNeighborParticleIds());
		m_nbrQueryTri->outNeighborIds()->connect(m_pbdModule2->inNeighborTriangleIds());
		this->inTriangleIndex()->connect(m_pbdModule2->inTriangleIndex());
		this->inTriangleVertex()->connect(m_pbdModule2->inTriangleVertex());
		this->pushModule(m_pbdModule2);

		auto m_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(m_integrator->inTimeStep());
		this->inPosition()->connect(m_integrator->inPosition());
		this->inVelocity()->connect(m_integrator->inVelocity());
		this->inForce()->connect(m_integrator->inForceDensity());
		this->pushModule(m_integrator);

		auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
		this->inTimeStep()->connect(m_visModule->inTimeStep());
		this->inPosition()->connect(m_visModule->inPosition());
		this->inVelocity()->connect(m_visModule->inVelocity());
		m_visModule->varViscosity()->setValue(Real(1));
		this->varSmoothingLength()->connect(m_visModule->inSmoothingLength());
		m_nbrQueryPoint->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->pushModule(m_visModule);

		//TODO:
		auto m_meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->inTimeStep()->connect(m_meshCollision->inTimeStep());
		this->inPosition()->connect(m_meshCollision->inPosition());
		this->inVelocity()->connect(m_meshCollision->inVelocity());
		this->inTriangleVertex()->connect(m_meshCollision->inTriangleVertex());
		this->inTriangleIndex()->connect(m_meshCollision->inTriangleIndex());
		m_nbrQueryTri->outNeighborIds()->connect(m_meshCollision->inTriangleNeighborIds());
		this->pushModule(m_meshCollision);
	}

	DEFINE_CLASS(SemiAnalyticalPositionBasedFluidModel);
}  // namespace PhysIKA