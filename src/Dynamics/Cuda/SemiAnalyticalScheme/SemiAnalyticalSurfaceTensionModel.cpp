#include "SemiAnalyticalSurfaceTensionModel.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/SummationDensity.h"
#include "ParticleSystem/Module/IterativeDensitySolver.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"
#include "TriangularMeshConstraint.h"
//#include "ParticleShifting.h"
#include "SemiAnalyticalParticleShifting.h"
#include "ComputeParticleAnisotropy.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalSurfaceTensionModel, TDataType)

	template<typename TDataType>
	SemiAnalyticalSurfaceTensionModel<TDataType>::SemiAnalyticalSurfaceTensionModel()
		: GroupModule()
	{
		this->varSmoothingLength()->setValue(Real(0.012));//0.006

		//integrator
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->inTimeStep()->connect(integrator->inTimeStep());
		this->inPosition()->connect(integrator->inPosition());
		this->inVelocity()->connect(integrator->inVelocity());
		this->inForceDensity()->connect(integrator->inForceDensity());
		this->pushModule(integrator);

		//neighbor query
		auto nbrQuery =std::make_shared<NeighborPointQuery<TDataType>>();
		this->varSmoothingLength()->connect(nbrQuery->inRadius());
		this->inPosition()->connect(nbrQuery->inPosition());
		this->pushModule(nbrQuery);

		//triangle neighbor
		auto nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		this->varSmoothingLength()->connect(nbrQueryTri->inRadius());
		this->inPosition()->connect(nbrQueryTri->inPosition());
		this->inTriangleVer()->connect(nbrQueryTri->inTriPosition());
		this->inTriangleInd()->connect(nbrQueryTri->inTriangles());
		this->pushModule(nbrQueryTri);

		//mesh collision
		auto meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->inTimeStep()->connect(meshCollision->inTimeStep());
		this->inPosition()->connect(meshCollision->inPosition());
		this->inVelocity()->connect(meshCollision->inVelocity());
		this->inTriangleVer()->connect(meshCollision->inTriangleVertex());
		this->inTriangleInd()->connect(meshCollision->inTriangleIndex());
		nbrQueryTri->outNeighborIds()->connect(meshCollision->inTriangleNeighborIds());
		this->pushModule(meshCollision);

		//viscosity
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(0.5));//0.5
		this->inTimeStep()->connect(viscosity->inTimeStep());
		this->varSmoothingLength()->connect(viscosity->inSmoothingLength());
		this->inPosition()->connect(viscosity->inPosition());
		this->inVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->pushModule(viscosity);

		//particle shifting
		auto pshiftModule = std::make_shared<SemiAnalyticalParticleShifting<TDataType>>();
		this->inTimeStep()->connect(pshiftModule->inTimeStep());
		this->inPosition()->connect(pshiftModule->inPosition());
		this->inVelocity()->connect(pshiftModule->inVelocity());
		nbrQuery->outNeighborIds()->connect(pshiftModule->inNeighborIds());
		this->inTriangleVer()->connect(pshiftModule->inTriangleVer());
		this->inTriangleInd()->connect(pshiftModule->inTriangleInd());
		this->inAttribute()->connect(pshiftModule->inAttribute());
		nbrQueryTri->outNeighborIds()->connect(pshiftModule->inNeighborTriIds());
		this->varSurfaceTension()->connect(pshiftModule->varSurfaceTension());
		this->varAdhesionIntensity()->connect(pshiftModule->varAdhesionIntensity());
		this->varRestDensity()->connect(pshiftModule->varRestDensity());
		this->pushModule(pshiftModule);
	}

	DEFINE_CLASS(SemiAnalyticalSurfaceTensionModel);
}