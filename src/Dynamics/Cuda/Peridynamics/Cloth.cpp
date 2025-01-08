#include "Cloth.h"
#include "Topology/TriangleSet.h"
#include "Topology/PointSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"

#include "Collision/NeighborPointQuery.h"

#include "Module/LinearElasticitySolver.h"
#include "Module/ProjectivePeridynamics.h"
#include "Module/FixedPoints.h"

#include "Auxiliary/DataSource.h"

#include "SharedFunc.h"
#include "TriangularSystem.h"

#include "GLPointVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"

namespace dyno
{
	IMPLEMENT_TCLASS(Cloth, TDataType)

	template<typename TDataType>
	Cloth<TDataType>::Cloth()
		: TriangularSystem<TDataType>()
	{
		this->varHorizon()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					this->stateHorizon()->setValue(this->varHorizon()->getValue());
				})
		);

		this->varHorizon()->setValue(0.0085);

		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());

		this->animationPipeline()->pushModule(integrator);

		auto elasticity = std::make_shared<LinearElasticitySolver<TDataType>>();
		this->stateHorizon()->connect(elasticity->inHorizon());
		this->stateTimeStep()->connect(elasticity->inTimeStep());
		this->stateRestPosition()->connect(elasticity->inX());
		this->statePosition()->connect(elasticity->inY());
		this->stateVelocity()->connect(elasticity->inVelocity());
		this->stateBonds()->connect(elasticity->inBonds());
		this->animationPipeline()->pushModule(elasticity);

		auto pointRenderer = std::make_shared<GLPointVisualModule>();
		pointRenderer->setColor(Color(1, 0.2, 1));
		pointRenderer->setColorMapMode(GLPointVisualModule::PER_OBJECT_SHADER);
		pointRenderer->varPointSize()->setValue(0.002f);
		this->stateTriangleSet()->connect(pointRenderer->inPointSet());
		this->stateVelocity()->connect(pointRenderer->inColor());

		this->graphicsPipeline()->pushModule(pointRenderer);
		this->setVisible(true);

		auto wireRenderer = std::make_shared<GLWireframeVisualModule>();
		wireRenderer->varBaseColor()->setValue(Color(1.0, 0.8, 0.8));
		wireRenderer->varRadius()->setValue(0.001f);
		wireRenderer->varRenderMode()->setCurrentKey(GLWireframeVisualModule::CYLINDER);
		this->stateTriangleSet()->connect(wireRenderer->inEdgeSet());
		this->graphicsPipeline()->pushModule(wireRenderer);

		auto surfaceRenderer = std::make_shared<GLSurfaceVisualModule>();
		this->stateTriangleSet()->connect(surfaceRenderer->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRenderer);
	}

	template<typename TDataType>
	Cloth<TDataType>::~Cloth()
	{
		
	}

	template<typename TDataType>
	void Cloth<TDataType>::resetStates()
	{
		auto input = this->inTriangleSet()->getDataPtr();

		auto ts = this->stateTriangleSet()->getDataPtr();

		ts->copyFrom(*input);

		DArrayList<int> nbr;
		ts->requestPointNeighbors(nbr);
		auto& pts = ts->getPoints();

		this->statePosition()->resize(pts.size());
		this->stateVelocity()->resize(pts.size());

		this->statePosition()->assign(pts);
		this->stateVelocity()->reset();

		if (this->stateBonds()->isEmpty()) {
			this->stateBonds()->allocate();
		}
		
		auto nbrPtr = this->stateBonds()->getDataPtr();

		this->stateOldPosition()->assign(this->statePosition()->getData());
		this->stateRestPosition()->assign(this->statePosition()->getData());

		constructRestShapeWithSelf(*nbrPtr, nbr, this->statePosition()->getData());

		nbr.clear();
	}

	template<typename TDataType>
	void Cloth<TDataType>::preUpdateStates()
	{
		auto& posOld = this->stateOldPosition()->getData();
		auto& posNew = this->statePosition()->getData();

		posOld.assign(posNew);
	}

	DEFINE_CLASS(Cloth);
}