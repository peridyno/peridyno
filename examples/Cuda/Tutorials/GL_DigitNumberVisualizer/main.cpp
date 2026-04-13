#pragma once

#include <UbiApp.h> 
#include <GLRenderEngine.h>

#include "Array/Array.h"
#include "Matrix.h"
#include "Node.h"

#include "SceneGraph.h"
#include "GLInstanceVisualModule.h"
#include "Topology/TriangleSet.h"


#include <Topology/TriangleSet.h>

#include <GLPointVisualModule.h>
#include <GLWireframeVisualModule.h>
#include <GLSurfaceVisualModule.h>

#include <GLDigitNumVisualModule.h>
using namespace std;

using namespace dyno;


    //3. 渲染数字模型
    //  创建一个可视化模块，将数字模型与实例化渲染结合起来。用于渲染顶点，示例如下

    class SurfaceMesh : public Node
    {
    public:
        SurfaceMesh() {

            // geometry
            std::shared_ptr<TriangleSet<DataType3f>> triSet = std::make_shared<TriangleSet<DataType3f>>();
            triSet->loadObjFile(getAssetPath() + "standard/standard_sphere.obj");
            triSet->update();
            this->stateTriangles()->setDataPtr(triSet);

            //Point visualizer
            auto pointRender = std::make_shared<GLPointVisualModule>();
            pointRender->varBaseColor()->setValue(Color(1.0f, 0.0f, 0.0));
            pointRender->varPointSize()->setValue(0.02f);
            this->stateTriangles()->connect(pointRender->inPointSet());
            this->graphicsPipeline()->pushModule(pointRender);

            //Wireframe visualizer
            auto edgeRender = std::make_shared<GLWireframeVisualModule>();
            edgeRender->varBaseColor()->setValue(Color(0, 1, 0));
            edgeRender->varRenderMode()->setCurrentKey(GLWireframeVisualModule::ARROW);
            edgeRender->varLineWidth()->setValue(3.f);
            this->stateTriangles()->connect(edgeRender->inEdgeSet());
            this->graphicsPipeline()->pushModule(edgeRender);

            ////Triangle visualizer
            //auto triRender = std::make_shared<GLSurfaceVisualModule>();
            //triRender->varBaseColor()->setValue(Color(0, 0, 1));
            //this->stateTriangles()->connect(triRender->inTriangleSet());
            //this->graphicsPipeline()->pushModule(triRender);


            //Digit Index visualizer
            auto digitRender = std::make_shared<GLDigitNumVisualModule>();
            digitRender->varBaseColor()->setValue(Color(1.0f, 0.0f, 0.0));
            digitRender->varDigitScale()->setValue(0.025f);;
            digitRender->varDigitOffset()->setValue(Vec2f(0.05f, 0.05f));
            this->stateTriangles()->connect(digitRender->inPointSet());
            this->graphicsPipeline()->pushModule(digitRender);
        };

    public:
        DEF_INSTANCE_STATE(TriangleSet<DataType3f>, Triangles, "Topology");
    };

    int main()
    {
        auto scn = std::make_shared<SceneGraph>();

        auto mesh = scn->addNode(std::make_shared<SurfaceMesh>());
       
        UbiApp window(GUIType::GUI_QT);
        window.setSceneGraph(scn);
        window.initialize(1024, 768);
        window.mainLoop();

        return 0;
    }