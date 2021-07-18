#include "RenderEngine.h"
// dyno
#include "SceneGraph.h"
#include "Action.h"

// GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/glad.h>

#include "Utility.h"
#include "ShadowMap.h"
#include "SSAO.h"
#include "RenderHelper.h"
#include "RenderTarget.h"

#include "module/GLVisualModule.h"

namespace dyno
{
	class RenderQueue : public Action
	{
	private:
		void process(Node* node) override
		{
			if (!node->isVisible())
				return;

			for (auto iter : node->graphicsPipeline()->activeModules())
			{
				auto m = dynamic_cast<GLVisualModule*>(iter);
				if (m && m->isVisible())
				{
					//m->update();
					modules.push_back(m);
				}
			}
		}
	public:
		std::vector<dyno::GLVisualModule*> modules;
	};

	RenderEngine::RenderEngine()
	{
		mRenderHelper = new RenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
		mSSAO = new SSAO;
	}

	RenderEngine::~RenderEngine()
	{
		delete mRenderHelper;
		delete mShadowMap;
		delete mSSAO;
	}

	void RenderEngine::initialize()
	{
		if (!gladLoadGL()) {
			printf("Failed to load OpenGL!");
			exit(-1);
		}

		// some basic opengl settings
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glDepthFunc(GL_LEQUAL);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		initUniformBuffers();

		mSSAO->initialize();
		mShadowMap->initialize();
		mRenderHelper->initialize();
	}


	void RenderEngine::initUniformBuffers()
	{
		// create uniform block for transform
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		// create uniform block for light
		mLightUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		gl::glCheckError();
	}

	void RenderEngine::draw(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams)
	{
		// gather visual modules
		RenderQueue renderQueue;
		// enqueue render content
		if ((scene != 0) && (scene->getRootNode() != 0))
		{
			scene->getRootNode()->traverseTopDown(&renderQueue);
		}

		// update shadow map
		mShadowMap->update(scene, rparams);
				
		// setup scene transform matrices
		struct
		{
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 projection;
			int width;
			int height;
		} sceneUniformBuffer;
		sceneUniformBuffer.model = glm::mat4(1);
		sceneUniformBuffer.view = rparams.view;
		sceneUniformBuffer.projection = rparams.proj;
		sceneUniformBuffer.width = target->width;
		sceneUniformBuffer.height = target->height;

		mTransformUBO.load(&sceneUniformBuffer, sizeof(sceneUniformBuffer));
		mTransformUBO.bindBufferBase(0);

		// setup light block
		RenderParams::Light light = rparams.light;
		light.mainLightDirection = glm::vec3(rparams.view * glm::vec4(light.mainLightDirection, 0));
		mLightUBO.load(&light, sizeof(light));
		mLightUBO.bindBufferBase(1);

		// begin rendering
		target->bind();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		Vec3f c0 = Vec3f(rparams.bgColor0.x, rparams.bgColor0.y, rparams.bgColor0.z);
		Vec3f c1 = Vec3f(rparams.bgColor1.x, rparams.bgColor1.y, rparams.bgColor1.z);
		mRenderHelper->drawBackground(c0, c1);

		glClear(GL_DEPTH_BUFFER_BIT);
		// draw a plane
		if (rparams.showGround)
		{
			mRenderHelper->drawGround(rparams.groudScale);
		}


		glBegin(GL_TRIANGLES);

		glVertex3f(0.0f, 1.0f, 0.0f); glColor3f(1.0f, 0.0f, 0.0f);

		glVertex3f(-1.0f, 0.0f, 0.0f); glColor3f(0.0f, 1.0f, 0.0f);

		glVertex3f(1.0f, 0.0f, 0.0f); glColor3f(0.0f, 0.0f, 1.0f);
		glEnd();

		// render modules
		for (GLVisualModule* m : renderQueue.modules)
		{
			m->paintGL(GLVisualModule::COLOR);
		}
		
		// draw scene bounding box
		if (rparams.showSceneBounds && scene != 0)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			mRenderHelper->drawBBox(p0, p1);
		}
		// draw axis
		if (rparams.showAxisHelper)
		{
			glViewport(10, 10, 100, 100);
			mRenderHelper->drawAxis();
		}
	}
}