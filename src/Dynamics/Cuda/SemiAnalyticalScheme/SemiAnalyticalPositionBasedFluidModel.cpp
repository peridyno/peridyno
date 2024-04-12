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

#include "SemiAnalyticalPBD.h"
#include "TriangularMeshConstraint.h"

#include "Auxiliary/DataSource.h"

namespace dyno {
	IMPLEMENT_TCLASS(SemiAnalyticalPositionBasedFluidModel, TDataType)

		template <typename TDataType>
	SemiAnalyticalPositionBasedFluidModel<TDataType>::SemiAnalyticalPositionBasedFluidModel()
		: GroupModule()
	{
		auto smoothingLength = std::make_shared<FloatingNumber<TDataType>>();
		smoothingLength->setName("Smoothing Length");
		smoothingLength->varValue()->setValue(Real(0.012));
		this->pushModule(smoothingLength);

		auto samplingDistance = std::make_shared<FloatingNumber<TDataType>>();
		samplingDistance->setName("Sampling Distance");
		samplingDistance->varValue()->setValue(Real(0.005));
		this->pushModule(samplingDistance);

		auto m_nbrQueryPoint = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(m_nbrQueryPoint->inRadius());
		this->inPosition()->connect(m_nbrQueryPoint->inPosition());
		this->pushModule(m_nbrQueryPoint);
		//m_nbrQueryPoint->initialize();

		auto m_nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		smoothingLength->outFloating()->connect(m_nbrQueryTri->inRadius());
		this->inPosition()->connect(m_nbrQueryTri->inPosition());
		this->inTriangleSet()->connect(m_nbrQueryTri->inTriangleSet());
		this->pushModule(m_nbrQueryTri);
		
// 		auto m_pbdModule2 = std::make_shared<SemiAnalyticalPBD<TDataType>>();
// 		this->inTimeStep()->connect(m_pbdModule2->inTimeStep());
// 		this->inPosition()->connect(m_pbdModule2->inPosition());
// 		this->inVelocity()->connect(m_pbdModule2->inVelocity());
// 		smoothingLength->outFloating()->connect(m_pbdModule2->inSmoothingLength());
// 		samplingDistance->outFloating()->connect(m_pbdModule2->inSamplingDistance());
// 		m_nbrQueryPoint->outNeighborIds()->connect(m_pbdModule2->inNeighborParticleIds());
// 		m_nbrQueryTri->outNeighborIds()->connect(m_pbdModule2->inNeighborTriangleIds());
// 		this->inTriangleSet()->connect(m_pbdModule2->inTriangleSet());
// 		this->pushModule(m_pbdModule2);

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
		smoothingLength->outFloating()->connect(m_visModule->inSmoothingLength());
		m_nbrQueryPoint->outNeighborIds()->connect(m_visModule->inNeighborIds());
		this->pushModule(m_visModule);

		//TODO:
		auto m_meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->inTimeStep()->connect(m_meshCollision->inTimeStep());
		this->inPosition()->connect(m_meshCollision->inPosition());
		this->inVelocity()->connect(m_meshCollision->inVelocity());
		this->inTriangleSet()->connect(m_meshCollision->inTriangleSet());
		m_nbrQueryTri->outNeighborIds()->connect(m_meshCollision->inTriangleNeighborIds());
		this->pushModule(m_meshCollision);

		this->varSmoothingLength()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					smoothingLength->varValue()->setValue(this->varSmoothingLength()->getValue());
				})
		);
	}

	DEFINE_CLASS(SemiAnalyticalPositionBasedFluidModel);
}  // namespace PhysIKA