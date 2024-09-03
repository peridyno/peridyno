#include <QtApp.h>
#include <SceneGraph.h>

#include "ObjIO/ObjLoader.h"

#include "SemiAnalyticalScheme/ParticleRelaxtionOnMesh.h"
#include <SemiAnalyticalScheme/TriangularMeshBoundary.h>
#include "ParticleSystem/SdfSampler.h"

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
	/*Creat a empty scene.*/
	std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	scn->setUpperBound(Vec3f(15.5, 15.0, 15.5));
	scn->setLowerBound(Vec3f(-15.5, -15.0, -15.5));

	/*Set up the gravity in the scene.*/
	scn->setGravity(Vec3f(0.0f));


	/*Load the shape file(.obj) file and render it:
	*	1. The first .obj is named as obj1;
	*	2. Set its scale, location;
	*	3. Render the .obj using SurfaceVisulModule;
	*	4. Set its color, visual metallic texture and Roughness.
	*/
	auto obj1 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj1->varScale()->setValue(Vec3f(0.7));
	obj1->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj1->varLocation()->setValue(Vec3f(0.0, 0, 0));
	auto SurfaceModule1 = obj1->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule1->setColor(Color(0.2));
	SurfaceModule1->setMetallic(1);
	SurfaceModule1->setRoughness(0.8);


	/*Convert the .obj(surface mesh) To the SDF(Signed distance field, Volume model)*/
	auto volume1 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume1->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj1->outTriangleSet()->promoteOuput()->connect(volume1->inTriangleSet()); /*Connect the .obj to the SDF convert module*/

	/* Load another .obj file as a hole and render it*/
	auto obj2 = scn->addNode(std::make_shared<ObjMesh<DataType3f>>());
	obj2->varScale()->setValue(Vec3f(0.45));
	obj2->varFileName()->setValue(getAssetPath() + "board/ball.obj");
	obj2->varLocation()->setValue(Vec3f(0, 0, 0));
	auto SurfaceModule2 = obj2->graphicsPipeline()->findFirstModule<GLSurfaceVisualModule>();
	SurfaceModule2->setColor(Color(0.2));
	SurfaceModule2->setMetallic(1);
	SurfaceModule2->setRoughness(0.8);

	/*Convert the .obj(surface mesh) To the SDF(Signed distance field, Volume model)*/
	auto volume2 = scn->addNode(std::make_shared<VolumeOctreeGenerator<DataType3f>>()); /*Creat a SDF convert module*/
	volume2->varSpacing()->setValue(0.005);	 /*The grid spacing of the SDF*/
	obj2->outTriangleSet()->promoteOuput()->connect(volume2->inTriangleSet());/*Connect the .obj to the SDF convert module*/

	/*Boolean operation: Two SDF. The shape(.obj1) sbtract the hole(.obj2)*/
	auto volume_Sphere_Uniform = scn->addNode(std::make_shared<VolumeOctreeBoolean<DataType3f>>());
	volume_Sphere_Uniform->varBooleanType()->getDataPtr()->setCurrentKey(VolumeOctreeBoolean<DataType3f>::SUBTRACTION_SET);
	/*SDF after Boolean operation*/
	volume1->connect(volume_Sphere_Uniform->importOctreeA()); /*B(.obj)-A(.obj)*/
	volume2->connect(volume_Sphere_Uniform->importOctreeB());

	/*Convert the SDF to particles*/
	auto Points = scn->addNode(std::make_shared<SdfSampler<DataType3f>>());
	Points->varSpacing()->setValue(0.01f);
	volume_Sphere_Uniform->connect(Points->importVolume());

	/*Render the particles*/
	auto pointVisual = std::make_shared<GLPointVisualModule>();
	Points->outPointSet()->promoteOuput()->connect(pointVisual->inPointSet());
	pointVisual->varPointSize()->setValue(0.007);
	pointVisual->varBaseColor()->setValue(Color(0.2, 0.2, 1));
	Points->graphicsPipeline()->pushModule(pointVisual);

	/*Generate a sphere*/
	auto meshes_1 = scn->addNode(std::make_shared<SphereModel<DataType3f>>());
	meshes_1->varLocation()->setValue(Vec3f(0.0, 0.0, 0.));
	meshes_1->varLatitude()->setValue(8);
	meshes_1->varScale()->setValue(Vec3f(0.6, 0.6, 0.6));
	meshes_1->varLongitude()->setValue(8);

	/*Generate points on meshes of the sphere*/
	auto pointset_1 = scn->addNode(std::make_shared<PointsBehindMesh<DataType3f>>());
	pointset_1->varSamplingDistance()->setValue(0.005);
	pointset_1->varThickness()->setValue(0.045);
	meshes_1->stateTriangleSet()->connect(pointset_1->inTriangleSet());
	
	/*Generate a cube*/
	auto meshes_2 = scn->addNode(std::make_shared<CubeModel<DataType3f>>());
	meshes_2->varLocation()->setValue(Vec3f(1.0, 0.0, 0.));
	meshes_2->varScale()->setValue(Vec3f(0.4, 0.4, 0.4));

	/*Generate points on meshes of the cube, and relax the point positions*/
	auto pointset_2 = scn->addNode(std::make_shared<ParticleRelaxtionOnMesh<DataType3f>>());
	pointset_2->varSamplingDistance()->setValue(0.005);
	pointset_2->varThickness()->setValue(0.045);
	meshes_2->stateTriangleSet()->connect(pointset_2->inTriangleSet());
	pointset_2->graphicsPipeline()->clear();

	auto ptRender = std::make_shared<GLPointVisualModule>();
	ptRender->setColor(Color(1, 0, 0));
	ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
	pointset_2->stateGhostPointSet()->connect(ptRender->inPointSet());
	pointset_2->graphicsPipeline()->pushModule(ptRender);

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


