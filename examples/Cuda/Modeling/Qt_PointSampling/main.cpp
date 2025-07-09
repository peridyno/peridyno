#include <QtApp.h>
#include <SceneGraph.h>

#include "ObjIO/ObjLoader.h"

#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"
#include "SemiAnalyticalScheme/TriangularMeshBoundary.h"

#include "Multiphysics/SdfSampler.h"
#include "Multiphysics/PoissonDiskSampler.h"
#include "Multiphysics/DevicePoissonDiskSampler.h"
#include "Samplers/PointsBehindMesh.h"
#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"

#include "BasicShapes/CubeModel.h"
#include "BasicShapes/SphereModel.h"
#include "BasicShapes/PlaneModel.h"

#include "Volume/VolumeOctreeGenerator.h"
#include "Volume/VolumeOctreeBoolean.h"
#include "Volume/VolumeGenerator.h"
#include "Volume/BasicShapeToVolume.h"

#include "GLSurfaceVisualModule.h"
#include "GLRenderEngine.h"
#include "GLPointVisualModule.h"

using namespace std;
using namespace dyno;

std::shared_ptr<SceneGraph> createScene()
{
	///*Creat a empty scene.*/
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(100.0, 100.0, 100.0));
	scn->setLowerBound(Vec3f(-100.0, -100.0, -100.0));
	scn->setGravity(Vec3f(0.0f));


	///*@brief Regular sampling in VolumeOctree*/
	auto obj1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj1->varScale()->setValue(Vec3f(0.7));
	obj1->varType()->setCurrentKey(1);
	obj1->varIcosahedronStep()->setValue(2);
	obj1->varLocation()->setValue(Vec3f(-1, 0.5, 0.0));
	auto volume1 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume1->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj1->stateTriangleSet()->promoteOuput()->connect(volume1->inTriangleSet()); /*Connect the .obj to the SDF convert module*/

	auto obj2 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj2->varScale()->setValue(Vec3f(0.45));
	obj2->varType()->setCurrentKey(1);
	obj2->varIcosahedronStep()->setValue(2);
	obj2->varLocation()->setValue(Vec3f(0.418-1,0.5,0));
	obj2->varType()->setCurrentKey(1);
	obj2->varIcosahedronStep()->setValue(2);
	obj2->setVisible(false);

	auto volume2 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume2->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj2->stateTriangleSet()->promoteOuput()->connect(volume2->inTriangleSet());/*Connect the .obj to the SDF convert module*/

	/*Boolean operation: Two SDF. The shape(.obj1) sbtract the hole(.obj2)*/
	auto volume_Sphere_Uniform = scn->addNode(std::make_shared<VolumeOctreeBoolean<DataType3f>>());
	volume_Sphere_Uniform->varBooleanType()->getDataPtr()->setCurrentKey(VolumeOctreeBoolean<DataType3f>::SUBTRACTION_SET);
	/*SDF after Boolean operation*/
	volume1->connect(volume_Sphere_Uniform->importOctreeA()); /*B(.obj)-A(.obj)*/
	volume2->connect(volume_Sphere_Uniform->importOctreeB());
	volume_Sphere_Uniform->graphicsPipeline()->disable();
    volume_Sphere_Uniform->animationPipeline()->disable();

	auto Points = scn->addNode(std::make_shared<SdfSampler<DataType3f>>());
	Points->varSpacing()->setValue(0.01f);
	volume_Sphere_Uniform->connect(Points->importVolumeOctree());

	auto pointVisual = std::make_shared<GLPointVisualModule>();
	Points->statePointSet()->promoteOuput()->connect(pointVisual->inPointSet());
	pointVisual->varPointSize()->setValue(0.007);
	pointVisual->varBaseColor()->setValue(Color(0.2, 0.2, 1));
	Points->graphicsPipeline()->pushModule(pointVisual);


	///*@brief Generate points on triangle meshes.*/
	auto meshes_1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	meshes_1->varLocation()->setValue(Vec3f(1.0, 0.5, 0.));
	meshes_1->varLatitude()->setValue(16);
	meshes_1->varLongitude()->setValue(12);
	meshes_1->varScale()->setValue(Vec3f(0.6, 0.6, 0.6));
	meshes_1->varType()->setCurrentKey(1);
	meshes_1->varIcosahedronStep()->setValue(2);
	meshes_1->setVisible(false);

	auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	pointset_1->varGeneratingDirection()->setValue(false);
	pointset_1->varSamplingDistance()->setValue(0.005);
	pointset_1->varThickness()->setValue(0.045);
	meshes_1->stateTriangleSet()->connect(pointset_1->inTriangleSet());

	///*@brief Generate points on meshes of the cube, and slightly shift the point positions.*/
	auto meshes_2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	meshes_2->varLocation()->setValue(Vec3f(2.0, 0.5, 0.));
	meshes_2->varScale()->setValue(Vec3f(0.4, 0.4, 0.4));
	auto pointset_2 = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset_2->varIterationNumber()->setValue(30);
	pointset_2->varSamplingDistance()->setValue(0.005);
	pointset_2->varThickness()->setValue(0.045);
	meshes_2->stateTriangleSet()->connect(pointset_2->inTriangleSet());
	pointset_2->graphicsPipeline()->clear();

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	pointset_2->statePointSet()->connect(ptRender->inPointSet());
	pointset_2->graphicsPipeline()->pushModule(ptRender);

	///*@brief Generate Poisson-disk distributed points in VolumeOctree on CPU*/
	auto obj3 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj3->varScale()->setValue(Vec3f(0.7));
	obj3->varType()->setCurrentKey(1);
	obj3->varIcosahedronStep()->setValue(2);
	obj3->varLocation()->setValue(Vec3f(0, 0.5, 0.0));
	
	auto volume3 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume3->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj3->stateTriangleSet()->promoteOuput()->connect(volume3->inTriangleSet()); /*Connect the .obj to the SDF convert module*/
	auto obj4 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj4->varScale()->setValue(Vec3f(0.45));
	obj4->varLocation()->setValue(Vec3f(0.35, 0.5, 0));
	obj4->varType()->setCurrentKey(1);
	obj4->varIcosahedronStep()->setValue(2);
	
	auto volume4 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume4->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj4->stateTriangleSet()->promoteOuput()->connect(volume4->inTriangleSet());/*Connect the .obj to the SDF convert module*/
	
	auto volume_bool2 = scn->addNode(std::make_shared<VolumeOctreeBoolean<DataType3f>>());
	volume_bool2->varBooleanType()->getDataPtr()->setCurrentKey(VolumeOctreeBoolean<DataType3f>::UNION_SET);
	volume3->connect(volume_bool2->importOctreeA()); /*B(.obj)-A(.obj)*/
	volume4->connect(volume_bool2->importOctreeB());
	volume_bool2->graphicsPipeline()->disable();
	auto poissonPointSet = scn->addNode(std::make_shared<PoissonDiskSampler<DataType3f>>());
	volume_bool2->connect(poissonPointSet->importVolumeOctree());
	//auto poissonPointSet = scn->addNode(std::make_shared<DevicePoissonDiskSampler<DataType3f>>());
	//poissonPointSet->varDelta()->setValue(0.001f);
	//poissonPointSet->varSpacing()->setValue(0.01f);
	//volume_bool2->connect(poissonPointSet->importVolumeOctree()); 

	///*@brief Generate points in Volume.*/
	auto obj5 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj5->varLocation()->setValue(Vec3f(0.0f, 1.2f, 0.0f));
	obj5->varRadius()->setValue(0.2f);
	auto volume5 = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	obj5->connect(volume5->importShape());
	auto Points5 = scn->addNode(std::make_shared<SdfSampler<DataType3f>>());
	Points5->varSpacing()->setValue(0.01f);
	volume5->connect(Points5->importVolume());

	///*@brief Generate Possion-disk distributed points in Volume on CPU.*/
	auto obj6 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj6->varRadius()->setValue(0.3f);
	obj6->varLocation()->setValue(Vec3f(0.75f, 1.2f, 0.0f));
	auto volume6 = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	obj6->connect(volume6->importShape());
	auto Points6 = scn->addNode(std::make_shared<PoissonDiskSampler<DataType3f>>());
	Points6->varSpacing()->setValue(0.01f);
	volume6->connect(Points6->importVolume());


	///*@brief Generate Possion-disk distributed points in Volume on GPU.*/
	auto obj7 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	obj7->varLocation()->setValue(Vec3f(1.6f, 1.2f, 0.0f));
	obj7->varRadius()->setValue(0.35);
	auto volume7 = scn->addNode(std::make_shared<BasicShapeToVolume<DataType3f>>());
	volume7->varGridSpacing()->setValue(0.01f);
	obj7->connect(volume7->importShape());
	auto Points7 = scn->addNode(std::make_shared<DevicePoissonDiskSampler<DataType3f>>());
	Points7->varDelta()->setValue(0.001f);
	Points7->varSpacing()->setValue(0.01f);
	volume7->connect(Points7->importVolume());

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
