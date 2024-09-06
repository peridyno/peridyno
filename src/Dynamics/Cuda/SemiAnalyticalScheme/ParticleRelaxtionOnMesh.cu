#include "ParticleRelaxtionOnMesh.h"
#include "Auxiliary/DataSource.h"
#include "Topology/PointSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleRelaxtionOnMesh, TDataType)

	template<typename TDataType>
	ParticleRelaxtionOnMesh<TDataType>::ParticleRelaxtionOnMesh()
		: ParticleSystem<TDataType>()
	{
		auto smoothingLength = std::make_shared<FloatingNumber<TDataType>>();
		smoothingLength->setName("Smoothing Length");
		smoothingLength->varValue()->setValue(Real(0.006));
		this->animationPipeline()->pushModule(smoothingLength);

		auto samplingDistance = std::make_shared<FloatingNumber<TDataType>>();
		samplingDistance->setName("Sampling Distance");
		samplingDistance->varValue()->setValue(Real(0.005));
		this->animationPipeline()->pushModule(samplingDistance);

		ptr_integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateDelta()->connect(ptr_integrator->inTimeStep());
		this->statePosition()->connect(ptr_integrator->inPosition());
		this->stateVelocity()->connect(ptr_integrator->inVelocity());
		this->stateForce()->connect(ptr_integrator->inForceDensity());
		//this->animationPipeline()->pushModule(ptr_integrator);

		ptr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(ptr_nbrQuery->inRadius());
		this->statePosition()->connect(ptr_nbrQuery->inPosition());
		//this->animationPipeline()->pushModule(ptr_nbrQuery);

		ptr_density = std::make_shared<IterativeDensitySolver<TDataType>>();
		smoothingLength->outFloating()->connect(ptr_density->inSmoothingLength());
		samplingDistance->outFloating()->connect(ptr_density->inSamplingDistance());
		this->stateDelta()->connect(ptr_density->inTimeStep());
		this->statePosition()->connect(ptr_density->inPosition());
		this->stateVelocity()->connect(ptr_density->inVelocity());
		ptr_nbrQuery->outNeighborIds()->connect(ptr_density->inNeighborIds());
		//this->animationPipeline()->pushModule(ptr_density);

		ptr_viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		ptr_viscosity->varViscosity()->setValue(Real(50.0));
		this->stateDelta()->connect(ptr_viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(ptr_viscosity->inSmoothingLength());
		this->statePosition()->connect(ptr_viscosity->inPosition());
		this->stateVelocity()->connect(ptr_viscosity->inVelocity());
		ptr_nbrQuery->outNeighborIds()->connect(ptr_viscosity->inNeighborIds());
		//this->animationPipeline()->pushModule(ptr_viscosity);

		/*
		* 
		*/
		auto triangleNeiborLength = std::make_shared<FloatingNumber<TDataType>>();
		triangleNeiborLength->setName("Triangle Neibor  Length");
		triangleNeiborLength->varValue()->setValue(Real(0.012));
		this->animationPipeline()->pushModule(triangleNeiborLength);


		ptr_nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		triangleNeiborLength->outFloating()->connect(ptr_nbrQueryTri->inRadius());
		this->statePosition()->connect(ptr_nbrQueryTri->inPosition());
		this->inTriangleSet()->connect(ptr_nbrQueryTri->inTriangleSet());
		//this->animationPipeline()->pushModule(ptr_nbrQueryTri);

		ptr_meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		ptr_meshCollision->varThickness()->setValue(0.003);
		this->stateDelta()->connect(ptr_meshCollision->inTimeStep());
		this->statePosition()->connect(ptr_meshCollision->inPosition());
		this->stateVelocity()->connect(ptr_meshCollision->inVelocity());
		this->inTriangleSet()->connect(ptr_meshCollision->inTriangleSet());
		ptr_nbrQueryTri->outNeighborIds()->connect(ptr_meshCollision->inTriangleNeighborIds());
		//this->animationPipeline()->pushModule(ptr_meshCollision);

		ptr_normalForce = std::make_shared<NormalForce<TDataType >>();
		this->stateDelta()->connect(ptr_normalForce->inTimeStep());
		this->inParticleNormal()->connect(ptr_normalForce->inParticleNormal());
		this->inTriangleSet()->connect(ptr_normalForce->inTriangleSet());
		this->statePosition()->connect(ptr_normalForce->inPosition());
		this->stateVelocity()->connect(ptr_normalForce->inVelocity());
		this->inParticleBelongTriangleIndex()->connect(ptr_normalForce->inParticleMeshID());
		ptr_nbrQueryTri->outNeighborIds()->connect(ptr_normalForce->inTriangleNeighborIds());
		//this->animationPipeline()->pushModule(ptr_normalForce);

	}


	template<typename TDataType>
	ParticleRelaxtionOnMesh<TDataType>::~ParticleRelaxtionOnMesh()
	{
		Log::sendMessage(Log::Info, "ParticleRelaxtionOnMesh released \n");
	}


	template<typename TDataType>
	void ParticleRelaxtionOnMesh<TDataType>::preUpdateStates()
	{

	}

	
	template<typename TDataType>
	void ParticleRelaxtionOnMesh<TDataType>::loadInitialStates()
	{
		int num = this->inPointSet()->getDataPtr()->getPointSize();
		if(num!=0)
		{

			this->statePosition()->resize(num);
			this->stateVelocity()->resize(num);
			this->stateVelocity()->reset();
			this->stateForce()->resize(num);
			this->stateForce()->reset();

			this->statePosition()->assign(this->inPointSet()->getDataPtr()->getPoints());
		}
		else {
			this->statePosition()->resize(0);
			this->stateVelocity()->resize(0);
			this->stateForce()->resize(0);
		}
	}


	template<typename TDataType>
	void  ParticleRelaxtionOnMesh<TDataType>::particleRelaxion()
	{
		std::cout << "Particle Relaxion on Mesh";

		//std::cout << this->stateTimeStep()->getValue() << std::endl;
		for (int i = 0; i < this->varIterationNumber()->getValue(); i++)
		{
			if (i % 5 == 0) std::cout << ".";
			ptr_nbrQuery->inRadius()->setValue(this->varPointNeighborLength()->getValue());
			ptr_nbrQuery->update();
			
			ptr_density->varIterationNumber()->setValue(this->varDensityIteration()->getValue());
			ptr_density->update();
			
			ptr_viscosity->varViscosity()->setValue(this->varViscosityStrength()->getValue());
			ptr_viscosity->update();
			
			
			ptr_integrator->update();

			ptr_nbrQueryTri->inRadius()->setValue(this->varMeshNeighborLength()->getValue());
			ptr_nbrQueryTri->update();

			ptr_meshCollision->varThickness()->setValue(this->varMeshCollisionThickness()->getValue());
			ptr_meshCollision->update();

			ptr_normalForce->varStrength()->setValue(this->varNormalForceStrength()->getValue());
			ptr_normalForce->update();
		}

		std::cout << std::endl;
		std::cout << "Particle Relaxion Finished." << std::endl;
	};

	template<typename TDataType>
	void ParticleRelaxtionOnMesh<TDataType>::resetStates()
	{
		loadInitialStates();

		this->particleRelaxion();

		if (!this->statePosition()->isEmpty())
		{
			auto points = this->statePointSet()->getDataPtr();
			points->setPoints(this->statePosition()->getData());
		}
		else
		{
			auto points = this->statePointSet()->getDataPtr();
			points->clear();
		}
	}


	DEFINE_CLASS(ParticleRelaxtionOnMesh);
}