#include "ParticleFluid.h"

//ParticleSystem
#include "Module/CalculateNorm.h"
#include "Module/ParticleIntegrator.h"
#include "Module/ImplicitViscosity.h"

#include "Module/SemiImplicitDensitySolver.h"
#include "Module/IterativeDensitySolver.h"
#include "Module/DivergenceFreeSphSolver.h"
#include "Module/ImplicitISPH.h"
#include "Module/VariationalApproximateProjection.h"
#include "DualParticle/DualParticleIsphModule.h"
#include "DualParticle/VirtualSpatiallyAdaptiveStrategy.h"
#include "DualParticle/VirtualFissionFusionStrategy.h"
#include "DualParticle/VirtualColocationStrategy.h"


#include "ParticleSystemHelper.h"

//Framework
#include "Auxiliary/DataSource.h"

//Collision
#include "Collision/NeighborPointQuery.h"

//Topology
#include "Topology/PointSet.h"

//Rendering
#include "ColorMapping.h"
#include "GLPointVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleFluid, TDataType)

	template<typename TDataType>
	ParticleFluid<TDataType>::ParticleFluid()
		: ParticleSystem<TDataType>()
	{
		this->varSamplingDistance()->setRange(0.001, 1);
		this->varSmoothingLength()->setRange(1, 3);

		auto callback = std::make_shared<FCallBackFunc>(
			[=]() {
				auto d = this->varSamplingDistance()->getValue();
				auto h = this->varSmoothingLength()->getValue();
				this->stateSamplingDistance()->setValue(d);
				this->stateSmoothingLength()->setValue(d * h);
			}
		);

		this->varSamplingDistance()->attach(callback);
		this->varSmoothingLength()->attach(callback);
		
		this->varSmoothingLength()->setValue(1.2);


		//declare a callback function for varSolverType();
		auto switchIncompressibilitySolver = std::make_shared <FCallBackFunc>(
			[&]() {
				auto setupSISPHSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<SemiImplicitDensitySolver<TDataType>>();
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto setupDFSPHSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<DivergenceFreeSphSolver<DataType3f>>();
					density->varDivergenceSolverDisabled()->setValue(true);
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto setupPBFSolver = [=] {
					this->animationPipeline()->clear();

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<IterativeDensitySolver<DataType3f>>();
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};

				auto setupIISPHSolver = [=] {
					this->animationPipeline()->clear();

					this->varSmoothingLength()->setValue(1.5);

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto density = std::make_shared<ImplicitISPH<DataType3f>>();
					this->stateSmoothingLength()->connect(density->inSmoothingLength());
					this->stateSamplingDistance()->connect(density->inSamplingDistance());
					this->stateTimeStep()->connect(density->inTimeStep());
					this->statePosition()->connect(density->inPosition());
					this->stateVelocity()->connect(density->inVelocity());
					nbrQuery->outNeighborIds()->connect(density->inNeighborIds());
					this->animationPipeline()->pushModule(density);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(density->importModules());
					density->connect(viscosity->importModules());
					};
				auto setupDPSolver = [=] {
					this->animationPipeline()->clear();

					this->varSmoothingLength()->setValue(2.4);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto vpGen = std::make_shared<VirtualSpatiallyAdaptiveStrategy<TDataType>>();
					this->statePosition()->connect(vpGen->inRPosition());
					vpGen->varSamplingDistance()->setValue(Real(0.005));		/**Virtual particle radius*/
					vpGen->varCandidatePointCount()->getDataPtr()->setCurrentKey(VirtualSpatiallyAdaptiveStrategy<TDataType>::neighbors_33);
					this->animationPipeline()->pushModule(vpGen);

					auto rv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(rv_nbrQuery->inRadius());
					this->statePosition()->connect(rv_nbrQuery->inOther());
					vpGen->outVirtualParticles()->connect(rv_nbrQuery->inPosition());
					this->animationPipeline()->pushModule(rv_nbrQuery);

					auto vr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(vr_nbrQuery->inRadius());
					this->statePosition()->connect(vr_nbrQuery->inPosition());
					vpGen->outVirtualParticles()->connect(vr_nbrQuery->inOther());
					this->animationPipeline()->pushModule(vr_nbrQuery);

					auto vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(vv_nbrQuery->inRadius());
					vpGen->outVirtualParticles()->connect(vv_nbrQuery->inPosition());
					this->animationPipeline()->pushModule(vv_nbrQuery);

					auto m_dualIsph = std::make_shared<DualParticleIsphModule<TDataType>>();
					this->stateSmoothingLength()->connect(m_dualIsph->inSmoothingLength());
					this->stateSamplingDistance()->connect(m_dualIsph->inSamplingDistance());
					this->stateTimeStep()->connect(m_dualIsph->inTimeStep());
					this->statePosition()->connect(m_dualIsph->inRPosition());
					vpGen->outVirtualParticles()->connect(m_dualIsph->inVPosition());
					this->stateVelocity()->connect(m_dualIsph->inVelocity());
					m_dualIsph->varResidualThreshold()->setValue(0.001f);
					nbrQuery->outNeighborIds()->connect(m_dualIsph->inNeighborIds());
					rv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inRVNeighborIds());
					vr_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVRNeighborIds());
					vv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVVNeighborIds());
					m_dualIsph->varWarmStart()->setValue(true);
					this->animationPipeline()->pushModule(m_dualIsph);

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
					m_visModule->varViscosity()->setValue(Real(0.5));
					this->stateTimeStep()->connect(m_visModule->inTimeStep());
					this->stateSamplingDistance()->connect(m_visModule->inSamplingDistance());
					this->stateSmoothingLength()->connect(m_visModule->inSmoothingLength());
					this->stateTimeStep()->connect(m_visModule->inTimeStep());
					this->statePosition()->connect(m_visModule->inPosition());
					this->stateVelocity()->connect(m_visModule->inVelocity());
					nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
					this->animationPipeline()->pushModule(m_visModule);
				};
				auto setupFissionDPSolver = [=] {
					this->animationPipeline()->clear();

					this->varSmoothingLength()->setValue(2.4);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto vpGen = std::make_shared<VirtualFissionFusionStrategy<TDataType>>();
					vpGen->varTransitionRegionThreshold()->setValue(0.01);
					this->statePosition()->connect(vpGen->inRPosition());
					this->stateVelocity()->connect(vpGen->inRVelocity());
					nbrQuery->outNeighborIds()->connect(vpGen->inNeighborIds());
					this->stateSmoothingLength()->connect(vpGen->inSmoothingLength());
					this->stateSamplingDistance()->connect(vpGen->inSamplingDistance());
					this->stateFrameNumber()->connect(vpGen->inFrameNumber());
					this->stateTimeStep()->connect(vpGen->inTimeStep());
					this->animationPipeline()->pushModule(vpGen);
					vpGen->varMinDist()->setValue(0.002);
					this->animationPipeline()->pushModule(vpGen);

					auto rv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(rv_nbrQuery->inRadius());
					this->statePosition()->connect(rv_nbrQuery->inOther());
					vpGen->outVirtualParticles()->connect(rv_nbrQuery->inPosition());
					this->animationPipeline()->pushModule(rv_nbrQuery);

					auto vr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(vr_nbrQuery->inRadius());
					this->statePosition()->connect(vr_nbrQuery->inPosition());
					vpGen->outVirtualParticles()->connect(vr_nbrQuery->inOther());
					this->animationPipeline()->pushModule(vr_nbrQuery);

					auto vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(vv_nbrQuery->inRadius());
					vpGen->outVirtualParticles()->connect(vv_nbrQuery->inPosition());
					this->animationPipeline()->pushModule(vv_nbrQuery);

					auto m_dualIsph = std::make_shared<DualParticleIsphModule<TDataType>>();
					this->stateSmoothingLength()->connect(m_dualIsph->inSmoothingLength());
					this->stateSamplingDistance()->connect(m_dualIsph->inSamplingDistance());
					this->stateTimeStep()->connect(m_dualIsph->inTimeStep());
					this->statePosition()->connect(m_dualIsph->inRPosition());
					vpGen->outVirtualParticles()->connect(m_dualIsph->inVPosition());
					this->stateVelocity()->connect(m_dualIsph->inVelocity());
					m_dualIsph->varResidualThreshold()->setValue(0.001f);
					nbrQuery->outNeighborIds()->connect(m_dualIsph->inNeighborIds());
					rv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inRVNeighborIds());
					vr_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVRNeighborIds());
					vv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVVNeighborIds());
					m_dualIsph->varWarmStart()->setValue(true);
					this->animationPipeline()->pushModule(m_dualIsph);

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
					m_visModule->varViscosity()->setValue(Real(0.5));
					this->stateTimeStep()->connect(m_visModule->inTimeStep());
					this->stateSamplingDistance()->connect(m_visModule->inSamplingDistance());
					this->stateSmoothingLength()->connect(m_visModule->inSmoothingLength());
					this->stateTimeStep()->connect(m_visModule->inTimeStep());
					this->statePosition()->connect(m_visModule->inPosition());
					this->stateVelocity()->connect(m_visModule->inVelocity());
					nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
					this->animationPipeline()->pushModule(m_visModule);
				};
				auto setupVSSPHSolver = [=] {
					this->animationPipeline()->clear();

					this->varSmoothingLength()->setValue(2.4);

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto isph = std::make_shared<VariationalApproximateProjection<DataType3f>>();
					this->stateSmoothingLength()->connect(isph->inSmoothingLength());
					this->stateSamplingDistance()->connect(isph->inSamplingDistance());
					this->stateTimeStep()->connect(isph->inTimeStep());
					this->statePosition()->connect(isph->inPosition());
					this->stateVelocity()->connect(isph->inVelocity());
					nbrQuery->outNeighborIds()->connect(isph->inNeighborIds());
					this->animationPipeline()->pushModule(isph);

					auto viscosity = std::make_shared<ImplicitViscosity<TDataType>>();
					viscosity->varViscosity()->setValue(Real(1.0));
					this->stateTimeStep()->connect(viscosity->inTimeStep());
					this->stateSmoothingLength()->connect(viscosity->inSmoothingLength());
					this->stateSamplingDistance()->connect(viscosity->inSamplingDistance());
					this->statePosition()->connect(viscosity->inPosition());
					this->stateVelocity()->connect(viscosity->inVelocity());
					nbrQuery->outNeighborIds()->connect(viscosity->inNeighborIds());
					this->animationPipeline()->pushModule(viscosity);

					integrator->connect(isph->importModules());
					isph->connect(viscosity->importModules());
				};
				auto setupISPHSolver = [=] {
					this->animationPipeline()->clear();

					this->varSmoothingLength()->setValue(2.4);

					auto nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(nbrQuery->inRadius());
					this->statePosition()->connect(nbrQuery->inPosition());
					this->animationPipeline()->pushModule(nbrQuery);

					auto vpGen = std::make_shared<VirtualColocationStrategy<TDataType>>();
					this->statePosition()->connect(vpGen->inRPosition());
					this->animationPipeline()->pushModule(vpGen);

					auto rv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(rv_nbrQuery->inRadius());
					this->statePosition()->connect(rv_nbrQuery->inOther());
					vpGen->outVirtualParticles()->connect(rv_nbrQuery->inPosition());
					this->animationPipeline()->pushModule(rv_nbrQuery);

					auto vr_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(vr_nbrQuery->inRadius());
					this->statePosition()->connect(vr_nbrQuery->inPosition());
					vpGen->outVirtualParticles()->connect(vr_nbrQuery->inOther());
					this->animationPipeline()->pushModule(vr_nbrQuery);

					auto vv_nbrQuery = std::make_shared<NeighborPointQuery<TDataType>>();
					this->stateSmoothingLength()->connect(vv_nbrQuery->inRadius());
					vpGen->outVirtualParticles()->connect(vv_nbrQuery->inPosition());
					this->animationPipeline()->pushModule(vv_nbrQuery);

					auto m_dualIsph = std::make_shared<DualParticleIsphModule<TDataType>>();
					this->stateSmoothingLength()->connect(m_dualIsph->inSmoothingLength());
					this->stateSamplingDistance()->connect(m_dualIsph->inSamplingDistance());
					this->stateTimeStep()->connect(m_dualIsph->inTimeStep());
					this->statePosition()->connect(m_dualIsph->inRPosition());
					vpGen->outVirtualParticles()->connect(m_dualIsph->inVPosition());
					this->stateVelocity()->connect(m_dualIsph->inVelocity());
					m_dualIsph->varResidualThreshold()->setValue(0.001f);
					nbrQuery->outNeighborIds()->connect(m_dualIsph->inNeighborIds());
					rv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inRVNeighborIds());
					vr_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVRNeighborIds());
					vv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVVNeighborIds());
					m_dualIsph->varWarmStart()->setValue(true);
					this->animationPipeline()->pushModule(m_dualIsph);

					auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
					this->stateTimeStep()->connect(integrator->inTimeStep());
					this->statePosition()->connect(integrator->inPosition());
					this->stateVelocity()->connect(integrator->inVelocity());
					this->animationPipeline()->pushModule(integrator);

					auto m_visModule = std::make_shared<ImplicitViscosity<TDataType>>();
					m_visModule->varViscosity()->setValue(Real(0.5));
					this->stateTimeStep()->connect(m_visModule->inTimeStep());
					this->stateSamplingDistance()->connect(m_visModule->inSamplingDistance());
					this->stateSmoothingLength()->connect(m_visModule->inSmoothingLength());
					this->stateTimeStep()->connect(m_visModule->inTimeStep());
					this->statePosition()->connect(m_visModule->inPosition());
					this->stateVelocity()->connect(m_visModule->inVelocity());
					nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
					this->animationPipeline()->pushModule(m_visModule);
				};


				auto k = this->varIncompressibilitySolver()->currentKey();
				switch (k)
				{
				case IncompressibilitySolver::SISPH:
					setupSISPHSolver();
					break;
				case IncompressibilitySolver::DFSPH:
					setupDFSPHSolver();
					break;
				case IncompressibilitySolver::PBF:
					setupPBFSolver();
					break;
				case IncompressibilitySolver::IISPH:
					setupIISPHSolver();
					break;
				case IncompressibilitySolver::DualParticle:
					setupDPSolver();
					break;
				case IncompressibilitySolver::FissionDP:
					setupFissionDPSolver();
					break;
				case IncompressibilitySolver::VSSPH:
					setupVSSPHSolver();
					break;
				case IncompressibilitySolver::ISPH:
					setupISPHSolver();
					break;
				default:
					break;
				}
			}
		);

		this->varIncompressibilitySolver()->attach(switchIncompressibilitySolver);
		this->varIncompressibilitySolver()->setCurrentKey(IncompressibilitySolver::SISPH);

		//Setup the default render modules
		auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
		this->stateVelocity()->connect(calculateNorm->inVec());
		this->graphicsPipeline()->pushModule(calculateNorm);

		auto colorMapper = std::make_shared<ColorMapping<DataType3f>>();
		colorMapper->varMax()->setValue(5.0f);
		calculateNorm->outNorm()->connect(colorMapper->inScalar());
		this->graphicsPipeline()->pushModule(colorMapper);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varPointSize()->setValue(0.0035f);
		ptRender->varBaseColor()->setValue(Color(1, 0, 0));
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);

		this->statePointSet()->connect(ptRender->inPointSet());
		colorMapper->outColor()->connect(ptRender->inColor());

		this->graphicsPipeline()->pushModule(ptRender);

		this->setDt(Real(0.001));
	}

	template<typename TDataType>
	ParticleFluid<TDataType>::~ParticleFluid()
	{
		Log::sendMessage(Log::Info, "ParticleFluid released \n");
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::preUpdateStates()
	{
		auto emitters = this->getParticleEmitters();

		int curNum = this->statePosition()->size();
		int totalNum = curNum;
		if (emitters.size() > 0)
		{
			for (int i = 0; i < emitters.size(); i++)
			{
				totalNum += emitters[i]->sizeOfParticles();
			}

			if (totalNum > curNum)
			{
				DArray<Coord> pBuf;
				DArray<Coord> vBuf;

				if (curNum > 0)
				{
					pBuf.assign(this->statePosition()->getData());
					vBuf.assign(this->stateVelocity()->getData());
				}

				this->statePosition()->resize(totalNum);
				this->stateVelocity()->resize(totalNum);

				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();

				//Assign attributes from initial states
				if (curNum > 0)
				{
					new_pos.assign(pBuf, curNum, 0, 0);
					new_vel.assign(vBuf, curNum, 0, 0);

					pBuf.clear();
					vBuf.clear();
				}

				//Assign attributes from emitters
				int offset = curNum;
				for (int i = 0; i < emitters.size(); i++)
				{
					int num = emitters[i]->sizeOfParticles();
					if (num > 0)
					{
						DArray<Coord>& points = emitters[i]->getPositions();
						DArray<Coord>& vels = emitters[i]->getVelocities();

						new_pos.assign(points, num, offset, 0);
						new_vel.assign(vels, num, offset, 0);

						offset += num;
					}
				}
			}
		}

		if (this->varReshuffleParticles()->getValue())
		{
			auto& pos = this->statePosition()->getData();
			auto& vel = this->stateVelocity()->getData();

			DArray<OcKey> morton(pos.size());

			ParticleSystemHelper<TDataType>::calculateMortonCode(morton, pos, Real(0.005));
			ParticleSystemHelper<TDataType>::reorderParticles(pos, vel, morton);

			morton.clear();
		}
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::loadInitialStates()
	{
		auto initials = this->getInitialStates();

		if (initials.size() > 0) 
		{
			int totalNum = 0;

			for (int i = 0; i < initials.size(); i++)
			{
				totalNum += initials[i]->statePosition()->size();
			}

			this->statePosition()->resize(totalNum);
			this->stateVelocity()->resize(totalNum);

			if (totalNum > 0)
			{
				DArray<Coord>& new_pos = this->statePosition()->getData();
				DArray<Coord>& new_vel = this->stateVelocity()->getData();

				int offset = 0;
				for (int i = 0; i < initials.size(); i++)
				{
					auto inPos = initials[i]->statePosition()->getDataPtr();
					auto inVel = initials[i]->stateVelocity()->getDataPtr();
					if (!inPos->isEmpty())
					{
						uint num = inPos->size();

						new_pos.assign(*inPos, num, offset, 0);
						new_vel.assign(*inVel, num, offset, 0);

						offset += num;
					}
				}
			}
		}
		else {
			this->statePosition()->resize(0);
			this->stateVelocity()->resize(0);
		}
	}

	template<typename TDataType>
	void ParticleFluid<TDataType>::resetStates()
	{
		loadInitialStates();

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

	template<typename TDataType>
	void ParticleFluid<TDataType>::reshuffleParticles()
	{

	}

	DEFINE_CLASS(ParticleFluid);
}