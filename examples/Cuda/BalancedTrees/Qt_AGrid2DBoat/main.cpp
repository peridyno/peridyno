#include <QtApp.h>
#include <SceneGraph.h>

#include "HeightField/GLAdaptiveWaterVisualNode.h"
#include "RectangleRotateAroundAxis.h"
#include "GltfLoader.h"
#include "Volume/AdaptiveVolumeFromBasicShape2D.h"
#include "Volume/GLAdaptiveGridVisualNode2D.h"
#include "GLRotateAroundVisualNode.h"
#include "CWAaddBoatSpeed.h"

using namespace std;
using namespace dyno;

 std::shared_ptr<SceneGraph> createScene()
 {
	 std::shared_ptr<SceneGraph> scn = std::make_shared<SceneGraph>();
	 scn->setUpperBound(Vec3f(2, 2, 2));
	 scn->setLowerBound(Vec3f(-2, -2, -2));

	 auto boat = scn->addNode(std::make_shared<GltfLoader<DataType3f>>());
	 boat->varFileName()->setValue(getAssetPath() + "gltf/SailBoat/SailBoat.gltf");
	 boat->setVisible(false);

	 auto rectangle1 = scn->addNode(std::make_shared<RectangleRotateAroundAxis<DataType3f>>());
	 rectangle1->setForceUpdate(true);
	 boat->stateTextureMesh()->connect(rectangle1->inTextureMesh());
	 rectangle1->varInitialAngle()->setValue(0.0f);
	 rectangle1->varRotationRadius()->setValue(42.0f);
	 rectangle1->varFrequency()->setValue(1000);

	 auto rectangle2 = scn->addNode(std::make_shared<RectangleRotateAroundAxis<DataType3f>>());
	 rectangle2->setForceUpdate(true);
	 boat->stateTextureMesh()->connect(rectangle2->inTextureMesh());
	 rectangle2->varInitialAngle()->setValue(120.0f);
	 rectangle2->varRotationRadius()->setValue(42.0f);
	 rectangle2->varFrequency()->setValue(1000);

	 auto rectangle3 = scn->addNode(std::make_shared<RectangleRotateAroundAxis<DataType3f>>());
	 rectangle3->setForceUpdate(true);
	 boat->stateTextureMesh()->connect(rectangle3->inTextureMesh());
	 rectangle3->varInitialAngle()->setValue(240.0f);
	 rectangle3->varRotationRadius()->setValue(42.0f);
	 rectangle3->varFrequency()->setValue(1000);

	 auto rectangle4 = scn->addNode(std::make_shared<RectangleRotateAroundAxis<DataType3f>>());
	 rectangle4->setForceUpdate(true);
	 boat->stateTextureMesh()->connect(rectangle4->inTextureMesh());
	 rectangle4->varInitialAngle()->setValue(0.0f);
	 rectangle4->varRotationRadius()->setValue(21.0f);
	 rectangle4->varFrequency()->setValue(500);

	 auto rectangle5 = scn->addNode(std::make_shared<RectangleRotateAroundAxis<DataType3f>>());
	 rectangle5->setForceUpdate(true);
	 boat->stateTextureMesh()->connect(rectangle5->inTextureMesh());
	 rectangle5->varInitialAngle()->setValue(120.0f);
	 rectangle5->varRotationRadius()->setValue(21.0f);
	 rectangle5->varFrequency()->setValue(500);

	 auto rectangle6 = scn->addNode(std::make_shared<RectangleRotateAroundAxis<DataType3f>>());
	 rectangle6->setForceUpdate(true);
	 boat->stateTextureMesh()->connect(rectangle6->inTextureMesh());
	 rectangle6->varInitialAngle()->setValue(240.0f);
	 rectangle6->varRotationRadius()->setValue(21.0f);
	 rectangle6->varFrequency()->setValue(500);

	 auto AGrid = scn->addNode(std::make_shared<AdaptiveVolumeFromBasicShape2D<DataType3f>>());
	 AGrid->varLowerBound()->setValue(Vec2f(-100.0f, -100.0f));
	 AGrid->varUpperBound()->setValue(Vec2f(100.0f, 100.0f));
	 AGrid->varDx()->setValue(0.2);
	 AGrid->varLevelNum()->setValue(3);
	 AGrid->varIsHollow()->setValue(false);
	 AGrid->varDynamicMode()->setValue(true);
	 rectangle1->connect(AGrid->importShapes());
	 rectangle2->connect(AGrid->importShapes());
	 rectangle3->connect(AGrid->importShapes());
	 rectangle4->connect(AGrid->importShapes());
	 rectangle5->connect(AGrid->importShapes());
	 rectangle6->connect(AGrid->importShapes());

	 auto qtVisualizer = scn->addNode(std::make_shared<GLAdaptiveGridVisualNode2D<DataType3f>>());
	 AGrid->stateAGridSet()->connect(qtVisualizer->inAdaptiveVolume());
	 qtVisualizer->varEType()->setCurrentKey(GLAdaptiveGridVisualNode2D<DataType3f>::Quadtree_Edge);
	 qtVisualizer->varPPlane()->setCurrentKey(GLAdaptiveGridVisualNode2D<DataType3f>::XZ_Plane);

	 auto boatVisual = scn->addNode(std::make_shared<GLRotateAroundVisualNode<DataType3f>>());
	 boat->stateTextureMesh()->connect(boatVisual->stateTextureMesh());
	 rectangle1->connect(boatVisual->importShapes());
	 rectangle2->connect(boatVisual->importShapes());
	 rectangle3->connect(boatVisual->importShapes());
	 rectangle4->connect(boatVisual->importShapes());
	 rectangle5->connect(boatVisual->importShapes());
	 rectangle6->connect(boatVisual->importShapes());

	 auto wave = scn->addNode(std::make_shared<CWAaddBoatSpeed<DataType3f>>());
	 AGrid->stateAGridSet()->connect(wave->inAGrid2D());
	 wave->varWaterLevel()->setValue(8.0f);
	 rectangle1->connect(wave->importShapes());
	 rectangle2->connect(wave->importShapes());
	 rectangle3->connect(wave->importShapes());
	 rectangle4->connect(wave->importShapes());
	 rectangle5->connect(wave->importShapes());
	 rectangle6->connect(wave->importShapes());

	 auto qtVisualizer2 = scn->addNode(std::make_shared<GLAdaptiveWaterVisualNode<DataType3f>>());
	 qtVisualizer2->varWaterOffset()->setValue(-8.3f);
	 //AGrid->stateDecreaseMorton()->connect(qtVisualizer2->stateSeedMorton());
	 AGrid->stateAGridSet()->connect(qtVisualizer2->inAGridSet());
	 wave->stateHeigh()->connect(qtVisualizer2->inGrid());

	 return scn;
 }

int main()
{
	QtApp window;

	window.setSceneGraph(createScene());
	window.initialize(1280, 768);
	window.renderWindow()->getCamera()->setUnitScale(10.0);
	window.mainLoop();

	return 0;
}


