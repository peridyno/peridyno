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

		// for light transform... temporary
		mShadowMapUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);

		// create uniform block for light
		mLightUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mLightUBO.bindBufferBase(1);

		glCheckError();
	}

	void RenderEngine::renderSetup(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams)
	{
		// uniform block for transform matrices
		struct
		{
			// MVP
			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 projection;
			int width;
			int height;

		} sceneUniformBuffer, shadowUniformBuffer;

		sceneUniformBuffer.model = shadowUniformBuffer.model = glm::mat4(1);
		sceneUniformBuffer.view = rparams.view;
		sceneUniformBuffer.projection = rparams.proj;
		sceneUniformBuffer.width = target->width;
		sceneUniformBuffer.height = target->height;

		// set mvp transform
		mTransformUBO.load(&sceneUniformBuffer, sizeof(sceneUniformBuffer));

		glm::vec3 center = glm::vec3(0.f);
		float	  radius = 1.f;
		// update camera
		if (scene)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			glm::vec3 pmin = { p0[0], p0[1], p0[2] };
			glm::vec3 pmax = { p1[0], p1[1], p1[2] };
			center = (pmin + pmax) * 0.5f;
			radius = glm::distance(pmin, pmax) * 0.5f;
		}

		// main light MVP matrices	
		shadowUniformBuffer.projection = glm::ortho(-radius, radius, -radius, radius, -radius, radius);
		glm::vec3 lightUp = glm::vec3(0, 1, 0);
		if (glm::length(glm::cross(lightUp, rparams.light.mainLightDirection)) == 0.f)
		{
			lightUp = glm::vec3(0, 0, 1);
		}
		shadowUniformBuffer.view = glm::lookAt(center, center - rparams.light.mainLightDirection, lightUp);
		shadowUniformBuffer.width = mShadowMap->width;
		shadowUniformBuffer.height = mShadowMap->height;

		mShadowMapUBO.load(&shadowUniformBuffer, sizeof(shadowUniformBuffer));

		// light properties, convert into camera space
		RenderParams::Light light = rparams.light;
		light.mainLightDirection = glm::vec3(sceneUniformBuffer.view * glm::vec4(light.mainLightDirection, 0));
		light.mainLightVP = shadowUniformBuffer.projection * shadowUniformBuffer.view * glm::inverse(sceneUniformBuffer.view);
		mLightUBO.load(&light, sizeof(light));
	}


	void RenderEngine::draw(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams)
	{
		// pre-rendering 
		renderSetup(scene, target, rparams);
		
		// gather visual modules
		RenderQueue renderQueue;
		// enqueue render content
		if ((scene != 0) && (scene->getRootNode() != 0))
		{
			scene->getRootNode()->traverseTopDown(&renderQueue);
		}

		// render shadow map
		mShadowMapUBO.bindBufferBase(0);
		mShadowMap->update(renderQueue.modules, rparams);
		
		// transform block
		mTransformUBO.bindBufferBase(0);

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