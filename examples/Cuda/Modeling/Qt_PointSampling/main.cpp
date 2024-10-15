#include <QtApp.h>
#include <SceneGraph.h>

#include "ObjIO/ObjLoader.h"

#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>
#include "ParticleSystem/SdfSampler.h"
#include "ParticleSystem/PoissonDiskSampling.h"

#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"
#include "BasicShapes/PlaneModel.h"
#include "Normal.h"
#include "Commands/PointsBehindMesh.h"
#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"

#include "Volume/VolumeOctreeGenerator.h"
#include "Volume/VolumeOctreeBoolean.h"
#include "Volume/VolumeGenerator.h"

#include "GLSurfaceVisualModule.h"
#include <GLRenderEngine.h>
#include <GLPointVisualModule.h>

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	///*Creat a empty scene.*/
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(15.5, 15.0, 15.5));
	scn->setLowerBound(Vec3f(-15.5, -15.0, -15.5));

	///*Set up the gravity in the scene.*/
	//scn->setGravity(Vec3f(0.0f));

	/*
	*@brief Regular sampling in .sdf
	*/
	auto obj1 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj1->varScale()->setValue(Vec3f(0.7));
	obj1->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj1->varLocation()->setValue(Vec3f(0.0, 0.5, 0.0));
	auto SurfaceModule1 = obj1->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule1->setColor(Color(0.2));
	SurfaceModule1->setMetallic(1);
	SurfaceModule1->setRoughness(0.8);

	auto volume1 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume1->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj1->outTriangleSet()->promoteOuput()->connect(volume1->inTriangleSet()); /*Connect the .obj to the SDF convert module*/

	auto obj2 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj2->varScale()->setValue(Vec3f(0.45));
	obj2->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj2->varLocation()->setValue(Vec3f(0.0, 0.5, 0.0));
	auto SurfaceModule2 = obj2->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule2->setColor(Color(0.2));
	SurfaceModule2->setMetallic(1);
	SurfaceModule2->setRoughness(0.8);

	auto volume2 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume2->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj2->outTriangleSet()->promoteOuput()->connect(volume2->inTriangleSet());/*Connect the .obj to the SDF convert module*/

	/*Boolean operation: Two SDF. The shape(.obj1) sbtract the hole(.obj2)*/
	auto volume_Sphere_Uniform = scn->addNode(std::make_shared<VolumeOctreeBoolean<DataType3f>>());
	volume_Sphere_Uniform->varBooleanType()->getDataPtr()->setCurrentKey(VolumeOctreeBoolean<DataType3f>::SUBTRACTION_SET);
	/*SDF after Boolean operation*/
	volume1->connect(volume_Sphere_Uniform->importOctreeA()); /*B(.obj)-A(.obj)*/
	volume2->connect(volume_Sphere_Uniform->importOctreeB());
	volume_Sphere_Uniform->graphicsPipeline()->disable();

	auto Points = scn->addNode(std::make_shared<SdfSampler<DataType3f>>());
	Points->varSpacing()->setValue(0.01f);
	volume_Sphere_Uniform->connect(Points->importVolume());

	auto pointVisual = std::make_shared<GLPointVisualModule>();
	Points->statePointSet()->promoteOuput()->connect(pointVisual->inPointSet());
	pointVisual->varPointSize()->setValue(0.007);
	pointVisual->varBaseColor()->setValue(Color(0.2, 0.2, 1));
	Points->graphicsPipeline()->pushModule(pointVisual);


	/*
	*@brief Generate particles on triangle meshes.
	*/
	auto meshes_1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	meshes_1->varLocation()->setValue(Vec3f(1.0, 0.5, 0.));
	meshes_1->varLatitude()->setValue(16);
	meshes_1->varLongitude()->setValue(12);
	meshes_1->varScale()->setValue(Vec3f(0.6, 0.6, 0.6));

	auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	pointset_1->varGeneratingDirection()->setValue(false);
	pointset_1->varSamplingDistance()->setValue(0.005);
	pointset_1->varThickness()->setValue(0.045);
	meshes_1->stateTriangleSet()->connect(pointset_1->inTriangleSet());

	/*
	*@brief Generate points on meshes of the cube, and slightly shift the point positions.
	*/
	auto meshes_2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	meshes_2->varLocation()->setValue(Vec3f(2.0, 0.5, 0.));
	meshes_2->varScale()->setValue(Vec3f(0.4, 0.4, 0.4));

	auto pointset_2 = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset_2->varIterationNumber()->setValue(80);
	pointset_2->varSamplingDistance()->setValue(0.005);
	pointset_2->varThickness()->setValue(0.045);
	meshes_2->stateTriangleSet()->connect(pointset_2->inTriangleSet());
	pointset_2->graphicsPipeline()->clear();

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	pointset_2->statePointSet()->connect(ptRender->inPointSet());
	pointset_2->graphicsPipeline()->pushModule(ptRender);

	/*
	*@brief Poisson disk sampling in .sdf
	*/
	auto obj3 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj3->varScale()->setValue(Vec3f(0.7));
	obj3->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj3->varLocation()->setValue(Vec3f(1.0, 0.5, 0.0));
	auto SurfaceModule4 = obj3->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule4->setColor(Color(0.2));
	SurfaceModule4->setMetallic(1);
	SurfaceModule4->setRoughness(0.8);

	auto volume3 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume3->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj3->outTriangleSet()->promoteOuput()->connect(volume3->inTriangleSet()); /*Connect the .obj to the SDF convert module*/

	auto obj4 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj4->varScale()->setValue(Vec3f(0.45));
	obj4->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj4->varLocation()->setValue(Vec3f(1.0, 0.5, 0.0));
	auto SurfaceModule5 = obj4->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule5->setColor(Color(0.2));
	SurfaceModule5->setMetallic(1);
	SurfaceModule5->setRoughness(0.8);

	auto volume4 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume4->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj4->outTriangleSet()->promoteOuput()->connect(volume4->inTriangleSet());/*Connect the .obj to the SDF convert module*/

	auto volume_bool2 = scn->addNode(std::make_shared<VolumeOctreeBoolean<DataType3f>>());
	volume_bool2->varBooleanType()->getDataPtr()->setCurrentKey(VolumeOctreeBoolean<DataType3f>::UNION_SET);
	volume3->connect(volume_bool2->importOctreeA()); /*B(.obj)-A(.obj)*/
	volume4->connect(volume_bool2->importOctreeB());
	volume_bool2->graphicsPipeline()->disable();

	auto poissonPointSet = scn->addNode(std::make_shared<PoissonDiskSampling<DataType3f>>());
	volume_bool2->connect(poissonPointSet->importVolume());

	return scn;

}

int main()
{
	QtApp window;
	window.setSceneGraph(createScene());
	window.initialize(1280, 768);
	window.mainLoop();

	return 0;
}
