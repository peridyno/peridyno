#include "SemiAnalyticalSFINode.h"
#include "SemiAnalyticalSharedFunc.h"

//#include "PBFM.h"
#include "SemiAnalyticalSurfaceTensionModel.h"
#include "SemiAnalyticalPositionBasedFluidModel.h"
#include "ParticleSystem/ParticleSystem.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"

#include "Collision/NeighborPointQuery.h"
#include "Collision/NeighborTriangleQuery.h"

#include "SemiAnalyticalParticleShifting.h"

#include "Auxiliary/DataSource.h"

namespace dyno
{
	IMPLEMENT_TCLASS(SemiAnalyticalSFINode, TDataType)

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::SemiAnalyticalSFINode()
		: Node()
	{
		auto smoothingLength = std::make_shared<FloatingNumber<TDataType>>();
		smoothingLength->varValue()->setValue(Real(0.012));
		this->animationPipeline()->pushModule(smoothingLength);

		//integrator
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateForceDensity()->connect(integrator->inForceDensity());
		this->animationPipeline()->pushModule(integrator);

		//neighbor query
		auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		smoothingLength->outFloating()->connect(nbrQuery->inRadius());
		this->statePosition()->connect(nbrQuery->inPosition());
		this->animationPipeline()->pushModule(nbrQuery);

		//triangle neighbor
		auto nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		smoothingLength->outFloating()->connect(nbrQueryTri->inRadius());
		this->statePosition()->connect(nbrQueryTri->inPosition());
		this->stateTriangleVertex()->connect(nbrQueryTri->inTriPosition());
		this->stateTriangleIndex()->connect(nbrQueryTri->inTriangles());
		this->animationPipeline()->pushModule(nbrQueryTri);

		//mesh collision
		auto meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->stateTimeStep()->connect(meshCollision->inTimeStep());
		this->statePosition()->connect(meshCollision->inPosition());
		this->stateVelocity()->connect(meshCollision->inVelocity());
		this->stateTriangleVertex()->connect(meshCollision->inTriangleVertex());
		this->stateTriangleIndex()->connect(meshCollision->inTriangleIndex());
		nbrQueryTri->outNeighborIds()->connect(meshCollision->inTriangleNeighborIds());
		this->animationPipeline()->pushModule(meshCollision);

		//viscosity
		auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
		viscosity->varViscosity()->setValue(Real(0.5));//0.5
		this->stateTimeStep()->connect(viscosity->inTimeStep());
		smoothingLength->outFloating()->connect(viscosity->inSmoothingLength());
		this->statePosition()->connect(viscosity->inPosition());
		this->stateVelocity()->connect(viscosity->inVelocity());
		nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
		this->animationPipeline()->pushModule(viscosity);

		//particle shifting
		auto pshiftModule = std::make_shared<SemiAnalyticalParticleShifting<TDataType>>();
		this->stateTimeStep()->connect(pshiftModule->inTimeStep());
		this->statePosition()->connect(pshiftModule->inPosition());
		this->stateVelocity()->connect(pshiftModule->inVelocity());
		nbrQuery->outNeighborIds()->connect(pshiftModule->inNeighborIds());
		this->stateTriangleVertex()->connect(pshiftModule->inTriangleVer());
		this->stateTriangleIndex()->connect(pshiftModule->inTriangleInd());
		this->stateAttribute()->connect(pshiftModule->inAttribute());
		nbrQueryTri->outNeighborIds()->connect(pshiftModule->inNeighborTriIds());
		this->animationPipeline()->pushModule(pshiftModule);

	}

	template<typename TDataType>
	SemiAnalyticalSFINode<TDataType>::~SemiAnalyticalSFINode()
	{
	
	}

	template<typename TDataType>
	bool SemiAnalyticalSFINode<TDataType>::validateInputs()
	{
		auto inBoundary = this->inTriangleSet()->getDataPtr();
		bool validateBoundary = inBoundary != nullptr && !inBoundary->isEmpty();

		bool ret = Node::validateInputs();

		return ret && validateBoundary;
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::resetStates()
	{
		if (this->varFast()->getData() == true)
		{
			this->animationPipeline()->clear();
			auto pbd = std::make_shared<SemiAnalyticalPositionBasedFluidModel<DataType3f>>();
			pbd->varSmoothingLength()->setValue(0.0085);

			this->animationPipeline()->clear();
			this->stateTimeStep()->connect(pbd->inTimeStep());
			this->statePosition()->connect(pbd->inPosition());
			this->stateVelocity()->connect(pbd->inVelocity());
			this->stateForceDensity()->connect(pbd->inForce());
			this->stateTriangleVertex()->connect(pbd->inTriangleVertex());
			this->stateTriangleIndex()->connect(pbd->inTriangleIndex());
			this->animationPipeline()->pushModule(pbd);
		}

		auto& particleSystems = this->getParticleSystems();

		//add particle system
		int pNum = 0;
		for (int i = 0; i < particleSystems.size(); i++) {
			pNum += particleSystems[i]->statePosition()->size();
		}

		this->statePosition()->resize(pNum);
		this->stateVelocity()->resize(pNum);
		this->stateForceDensity()->resize(pNum);
		this->stateAttribute()->resize(pNum);

		DArray<Coord>& allpoints = this->statePosition()->getData();
		DArray<Coord>& allvels = this->stateVelocity()->getData();
		DArray<Attribute>& allattrs = this->stateAttribute()->getData();

		int offset = 0;
		for (int i = 0; i < particleSystems.size(); i++)
		{
			DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
			DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();

			int num = points.size();
			bool fixx = false;

			
			allpoints.assign(points, num, 0, offset);
			allvels.assign(vels, num, 0, offset);

			offset += num;
		}
		
		SetupAttributesForSFI(allattrs);

		auto triSet = this->inTriangleSet()->getDataPtr();
		this->stateTriangleVertex()->assign(triSet->getPoints());
		this->stateTriangleIndex()->assign(triSet->getTriangles());
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::preUpdateStates()
	{
		auto& particleSystems = this->getParticleSystems();

		int cur_num = this->statePosition()->size();

		if (particleSystems.size() == 0)
			return;

		int new_num = 0;
		for (int i = 0; i < particleSystems.size(); i++)
		{
			new_num += particleSystems[i]->statePosition()->size();
		}

		if (new_num != cur_num)
		{
			this->statePosition()->resize(new_num);
			this->stateVelocity()->resize(new_num);
			this->stateForceDensity()->resize(new_num);
			this->stateAttribute()->resize(new_num);
		}

		if (this->statePosition()->size() <= 0)
			return;

		auto& new_pos = this->statePosition()->getData();
		auto& new_vel = this->stateVelocity()->getData();
		auto& new_force = this->stateForceDensity()->getData();
		auto& new_atti = this->stateAttribute()->getData();

		int offset = 0;
		for (int i = 0; i < particleSystems.size(); i++)//update particle system
		{
			DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
			DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();
			DArray<Coord>& forces = particleSystems[i]->stateForce()->getData();
			int num = points.size();

			new_pos.assign(points, num, offset);
			new_vel.assign(vels, num, offset);

			offset += num;
		}

		this->stateForceDensity()->reset();
		SetupAttributesForSFI(new_atti);
		
		auto triSet = this->inTriangleSet()->getDataPtr();
		this->stateTriangleVertex()->assign(triSet->getPoints());
		this->stateTriangleIndex()->assign(triSet->getTriangles());
	}

	template<typename TDataType>
	void SemiAnalyticalSFINode<TDataType>::postUpdateStates()
	{
		auto& particleSystems = this->getParticleSystems();

		if (particleSystems.size() <= 0 || this->stateVelocity()->size() <= 0)
			return;

		auto& new_pos = this->statePosition()->getData();
		auto& new_vel = this->stateVelocity()->getData();
		auto& new_force = this->stateForceDensity()->getData();
		
		uint offset = 0;
		for (int i = 0; i < particleSystems.size(); i++)//extend current particles
		{
			DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
			DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();
			DArray<Coord>& forces = particleSystems[i]->stateForce()->getData();
			
			int num = points.size();

			points.assign(new_pos, num, 0, offset);
			vels.assign(new_vel, num, 0, offset);
			forces.assign(new_force, num, 0, offset);

			offset += num;
		}
	}

	DEFINE_CLASS(SemiAnalyticalSFINode);
}