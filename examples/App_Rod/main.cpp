#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GlutGUI/GLApp.h"

#include "Framework/SceneGraph.h"
#include "Topology/PointSet.h"
#include "Framework/Log.h"

#include "ParticleSystem/ParticleRod.h"
#include "ParticleSystem/StaticBoundary.h"

#include "ParticleSystem/ParticleElastoplasticBody.h"
#include "ParticleSystem/ParticleElasticBody.h"
#include "ParticleSystem/StaticBoundary.h"
#include "ParticleSystem/ElasticityModule.h"
#include "RigidBody/RigidBody.h"
#include "SurfaceMeshRender.h"
#include "PointRenderModule.h"

using namespace std;
using namespace dyno;

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
//	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, false);

	scene.setLowerBound(Vector3f(0.0f, 0.0f, 0.0f));
	scene.setUpperBound(Vector3f(1.0f, 1.0f, 1.0f));

	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));

	//	std::shared_ptr<ParticleRod<DataType3f>> child3 = scene.createNewScene<ParticleRod<DataType3f>>("Rod");

	std::shared_ptr<ParticleRod<DataType3f>> child3 = std::make_shared<ParticleRod<DataType3f>>("Rod");
	root->addParticleSystem(child3);

	Vector3f CableStart = (0.00001f*Vector3f(15996, 3140, 19990) + 0.2f);
	Vector3f CableEnd = (0.00001f*Vector3f(15649, 6555, 19990) + 0.2f);

	child3->setMass(1.0);

	int numSegment = 10;
	int numPoint = numSegment + 1;

	Vector3f Cable = CableEnd - CableStart;

	child3->m_horizon.setValue(Cable.norm() / numSegment);
	child3->setMaterialStiffness(1.0);

	std::vector<Vector3f> particles;
	for (int i = 0; i < numPoint; i++)
	{
		Vector3f pi = CableStart + Cable * (float)i / numSegment;
		particles.push_back(pi);

		if (i == 0)
			child3->addFixedParticle(0, pi);

// 		if (i == numSegment)
// 			child3->setFixedParticle(i, pi);
	}

	child3->setParticles(particles);

	auto pointsRender = std::make_shared<PointRenderModule>();
	child3->addVisualModule(pointsRender);

	child3->setVisible(true);


	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");

	return 0;
}
