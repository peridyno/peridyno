#include <GlfwApp.h>
#include "SceneGraph.h"
#include <Log.h>

#include <Module/CalculateNorm.h>
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>
#include <ColorMapping.h>
#include <ImColorbar.h>
#include "DualParticleSystem/DualParticleFluid.h"
#include "ParticleSystem/MakeParticleSystem.h"
#include <BasicShapes/CubeModel.h>
#include <Samplers/CubeSampler.h>
#include <ParticleSystem/Emitters/SquareEmitter.h>
#include <StaticTriangularMesh.h>
#include <GLSurfaceVisualModule.h>
#include "Collision/Attribute.h"
#include "DualParticleSystem/Module/VirtualSpatiallyAdaptiveStrategy.h"
#include "DualParticleSystem/Module/VirtualColocationStrategy.h"
#include "DualParticleSystem/Module/VirtualParticleShiftingStrategy.h"
#include "DualParticleSystem/Module/DualParticleIsphModule.h"
#include "ParticleSystem/Module/ImplicitViscosity.h"
#include "RotatingSquarePatchModule.h"
#include "Auxiliary/DataSource.h"
using namespace std;
using namespace dyno;


std::shared_ptr<SceneGraph> createScene()
{
	std::shared_ptr<SceneGraph> scene = std::make_shared<SceneGraph>();
	scene->setGravity(Vec3f(0));
	scene->setUpperBound(Vec3f(3.0));
	scene->setLowerBound(Vec3f(-3.0));

	//Create a cube
	auto cube = scene->addNode(std::make_shared<CubeModel<DataType3f>>());
	cube->varLocation()->setValue(Vec3f(0.5, 0.5, 0.0));
	cube->varLength()->setValue(Vec3f(0.24, 0.001, 0.24));
	cube->graphicsPipeline()->disable();

	//Create a sampler
	auto sampler = scene->addNode(std::make_shared<CubeSampler<DataType3f>>());
	sampler->varSamplingDistance()->setValue(0.005);
	sampler->setVisible(false);
	cube->outCube()->connect(sampler->inCube());
	auto initialParticles = scene->addNode(std::make_shared<MakeParticleSystem<DataType3f>>());
	sampler->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());


	//auto ptsLoader = scene->addNode(std::make_shared<ParticleLoader<DataType3f>>());
	//ptsLoader->loadParticles(Vec3f(-0.12, 0.1, -0.12), Vec3f(0.12, 0.104, 0.12), 0.005);
	//auto initialParticles = scene->addNode(std::make_shared<MakeParticleSystem<DataType3f >>());
	//ptsLoader->statePointSet()->promoteOuput()->connect(initialParticles->inPoints());

	auto fluid = scene->addNode(std::make_shared<DualParticleFluid<DataType3f>>());
	initialParticles->connect(fluid->importInitialStates());

	fluid->animationPipeline()->clear();
	{
		fluid->varReshuffleParticles()->setValue(false);

		auto smoothingLength = std::make_shared<FloatingNumber<DataType3f>>();
		fluid->animationPipeline()->pushModule(smoothingLength);
		smoothingLength->varValue()->setValue(Real(0.012));

		auto samplingDistance = std::make_shared<FloatingNumber<DataType3f>>();
		fluid->animationPipeline()->pushModule(smoothingLength);
		samplingDistance->varValue()->setValue(Real(0.005));

		std::shared_ptr<VirtualParticleGenerator<DataType3f>> vpGen;

		if (fluid->varVirtualParticleSamplingStrategy()->getDataPtr()->currentKey()
			== DualParticleFluid<DataType3f>::SpatiallyAdaptiveStrategy)
		{
			auto m_adaptive_virtual_position = std::make_shared<VirtualSpatiallyAdaptiveStrategy<DataType3f>>();
			fluid->statePosition()->connect(m_adaptive_virtual_position->inRPosition());
			m_adaptive_virtual_position->varSamplingDistance()->setValue(Real(0.005));		/**Virtual particle radius*/
			m_adaptive_virtual_position->varCandidatePointCount()->getDataPtr()->setCurrentKey(VirtualSpatiallyAdaptiveStrategy<DataType3f>::neighbors_33);
			vpGen = m_adaptive_virtual_position;
		}
		else if (fluid->varVirtualParticleSamplingStrategy()->getDataPtr()->currentKey()
			== DualParticleFluid<DataType3f>::ParticleShiftingStrategy)
		{
			auto m_virtual_particle_shifting = std::make_shared<VirtualParticleShiftingStrategy<DataType3f >>();
			fluid->stateFrameNumber()->connect(m_virtual_particle_shifting->inFrameNumber());
			fluid->statePosition()->connect(m_virtual_particle_shifting->inRPosition());
			fluid->stateTimeStep()->connect(m_virtual_particle_shifting->inTimeStep());
			fluid->animationPipeline()->pushModule(m_virtual_particle_shifting);
			vpGen = m_virtual_particle_shifting;
		}
		else if (fluid->varVirtualParticleSamplingStrategy()->getDataPtr()->currentKey()
			== DualParticleFluid<DataType3f>::ColocationStrategy)
		{
			auto m_virtual_equal_to_Real = std::make_shared<VirtualColocationStrategy<DataType3f >>();
			fluid->statePosition()->connect(m_virtual_equal_to_Real->inRPosition());
			fluid->animationPipeline()->pushModule(m_virtual_equal_to_Real);
			vpGen = m_virtual_equal_to_Real;
		}


		fluid->animationPipeline()->pushModule(vpGen);
		vpGen->outVirtualParticles()->connect(fluid->stateVirtualPosition());

		auto m_nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(m_nbrQuery->inRadius());
		fluid->statePosition()->connect(m_nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(m_nbrQuery);

		auto m_rv_nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(m_rv_nbrQuery->inRadius());
		fluid->statePosition()->connect(m_rv_nbrQuery->inOther());
		vpGen->outVirtualParticles()->connect(m_rv_nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(m_rv_nbrQuery);

		auto m_vr_nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(m_vr_nbrQuery->inRadius());
		fluid->statePosition()->connect(m_vr_nbrQuery->inPosition());
		vpGen->outVirtualParticles()->connect(m_vr_nbrQuery->inOther());
		fluid->animationPipeline()->pushModule(m_vr_nbrQuery);

		auto m_vv_nbrQuery = std::make_shared<NeighborPointQuery<DataType3f>>();
		smoothingLength->outFloating()->connect(m_vv_nbrQuery->inRadius());
		vpGen->outVirtualParticles()->connect(m_vv_nbrQuery->inPosition());
		fluid->animationPipeline()->pushModule(m_vv_nbrQuery);

		auto m_dualIsph = std::make_shared<DualParticleIsphModule<DataType3f>>();
		smoothingLength->outFloating()->connect(m_dualIsph->varSmoothingLength());
		fluid->stateTimeStep()->connect(m_dualIsph->inTimeStep());
		fluid->statePosition()->connect(m_dualIsph->inRPosition());
		vpGen->outVirtualParticles()->connect(m_dualIsph->inVPosition());
		fluid->stateVelocity()->connect(m_dualIsph->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_dualIsph->inNeighborIds());
		m_rv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inRVNeighborIds());
		m_vr_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVRNeighborIds());
		m_vv_nbrQuery->outNeighborIds()->connect(m_dualIsph->inVVNeighborIds());
		fluid->stateTimeStep()->connect(m_dualIsph->inTimeStep());
		fluid->animationPipeline()->pushModule(m_dualIsph);

		auto m_integrator = std::make_shared<RotatingSquarePatchModule<DataType3f>>();
		fluid->stateFrameNumber()->connect(m_integrator->inFrameNumber());
		fluid->stateTimeStep()->connect(m_integrator->inTimeStep());
		fluid->statePosition()->connect(m_integrator->inPosition());
		fluid->stateVelocity()->connect(m_integrator->inVelocity());
		fluid->stateParticleAttribute()->connect(m_integrator->inAttribute());
		fluid->animationPipeline()->pushModule(m_integrator);

		auto m_visModule = std::make_shared<ImplicitViscosity<DataType3f>>();
		m_visModule->varViscosity()->setValue(Real(0.1));
		fluid->stateTimeStep()->connect(m_visModule->inTimeStep());
		smoothingLength->outFloating()->connect(m_visModule->inSmoothingLength());
		fluid->stateTimeStep()->connect(m_visModule->inTimeStep());
		fluid->statePosition()->connect(m_visModule->inPosition());
		fluid->stateVelocity()->connect(m_visModule->inVelocity());
		m_nbrQuery->outNeighborIds()->connect(m_visModule->inNeighborIds());
		fluid->animationPipeline()->pushModule(m_visModule);
	
	}

	auto calculateNorm = std::make_shared<CalculateNorm<DataType3f>>();
	auto colorMapper = std::make_shared<ColorMapping<DataType3f >>();
	colorMapper->varMax()->setValue(5.0f);

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	ptRender->varPointSize()->setValue(0.004);

	fluid->stateVelocity()->connect(calculateNorm->inVec());
	fluid->statePointSet()->connect(ptRender->inPointSet());

	calculateNorm->outNorm()->connect(colorMapper->inScalar());
	colorMapper->outColor()->connect(ptRender->inColor());

	fluid->graphicsPipeline()->pushModule(calculateNorm);
	fluid->graphicsPipeline()->pushModule(colorMapper);
	fluid->graphicsPipeline()->pushModule(ptRender);

	auto vpRender = std::make_shared<GLPointVisualModule>();
	vpRender->setColor(Color(1, 1, 0));
	vpRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	fluid->stateVirtualPointSet()->connect(vpRender->inPointSet());
	vpRender->varPointSize()->setValue(0.0005);
	fluid->graphicsPipeline()->pushModule(vpRender);

	// A simple color bar widget for node
	auto colorBar = std::make_shared<ImColorbar>();
	colorBar->varMax()->setValue(5.0f);
	calculateNorm->outNorm()->connect(colorBar->inScalar());
	// add the widget to app
	fluid->graphicsPipeline()->pushModule(colorBar);

	return scene;
}

int main()
{
	GlfwApp window;
	window.setSceneGraph(createScene());
	window.initialize(1024, 768);
	window.mainLoop();

	return 0;
}




