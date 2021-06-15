#include "RenderEngine.h"
// dyno
#include "Framework/SceneGraph.h"
#include "Action/Action.h"
// GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/glad.h>
#include "Utility.h"
#include "ShadowMap.h"
#include "RenderHelper.h"
#include "RenderTarget.h"

#include "module/GLVisualModule.h"
#include "module/SurfaceRender.h"
#include "module/PointRender.h"

#include <random>

namespace dyno
{
	class DrawAct2 : public Action
	{
	public:
		DrawAct2(RenderEngine* engine)
		{
			mEngine = engine;
		}

	private:
		void process(Node* node) override
		{
			if (!node->isVisible())
			{
				return;
			}

			for (auto iter : node->getVisualModuleList())
			{
				auto m = std::dynamic_pointer_cast<GLVisualModule>(iter);
				if (m && m->isVisible())
				{
					m->updateRenderingContext();
					mEngine->enqueue(m.get());
				}
			}
		}
	private:
		RenderEngine* mEngine;
	};

	RenderEngine::RenderEngine()
	{
		mRenderHelper = new RenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
	}

	RenderEngine::~RenderEngine()
	{

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

		mShadowMap->initialize();
		mRenderHelper->initialize();

		// create a quad object
		mScreenQuad = GLMesh::ScreenQuad();

		// shader programs
		mSSAOProgram = CreateShaderProgram("screen.vert", "ssao.frag");
		
		// background
		mBackgroundProgram = CreateShaderProgram("screen.vert", "background.frag");
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
		
		// SSAO kernel
		mSSAOKernelUBO.create(GL_UNIFORM_BUFFER, GL_STATIC_DRAW);
		mSSAOKernelUBO.bindBufferBase(3);

		std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
		std::default_random_engine generator;
		std::vector<glm::vec3> ssaoKernel;
		for (unsigned int i = 0; i < 64; ++i)
		{
			glm::vec3 sample(
				randomFloats(generator) * 2.0 - 1.0,
				randomFloats(generator) * 2.0 - 1.0,
				randomFloats(generator)
			);
			sample = glm::normalize(sample);
			//sample *= randomFloats(generator);
			//ssaoKernel.push_back(sample);
			float scale = (float)i / 64.0;
			//scale = lerp(0.1f, 1.0f, scale * scale);
			scale = 0.1f + scale * scale * 0.9f;
			sample *= scale;
			ssaoKernel.push_back(sample);
		}

		mSSAOKernelUBO.load(ssaoKernel.data(), ssaoKernel.size() * sizeof(glm::vec3));

		// create SSAO noise here...
		std::vector<glm::vec3> ssaoNoise;
		for (unsigned int i = 0; i < 16; i++)
		{
			glm::vec3 noise(
				randomFloats(generator) * 2.0 - 1.0,
				randomFloats(generator) * 2.0 - 1.0,
				0.0f);
			ssaoNoise.push_back(noise);
		}

		mSSAONoiseTex.format = GL_RGB;
		mSSAONoiseTex.internalFormat = GL_RGB32F;
		mSSAONoiseTex.wrapS = GL_REPEAT;
		mSSAONoiseTex.wrapT = GL_REPEAT;
		mSSAONoiseTex.create();
		mSSAONoiseTex.load(4, 4, &ssaoNoise[0]);

		glCheckError();
	}

	void RenderEngine::renderSetup(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams)
	{
		// reset render queue...
		mRenderQueue.clear();
		// enqueue render content
		if (scene != 0)
		{
			if (scene->getRootNode() == nullptr)
			{
				return;
			}

			if (!scene->isInitialized())
				scene->initialize();

			scene->getRootNode()->traverseTopDown<DrawAct2>(this);
		}

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
		// preserve current framebuffer
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// pre-rendering 
		renderSetup(scene, target, rparams);
		updateShadowMap(rparams);
		
		// transform block
		mTransformUBO.bindBufferBase(0);

		target->drawColorTex();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		mShadowMap->bindShadowTex();

		// render background
		mBackgroundProgram.use();
		mBackgroundProgram.setVec3("uColor0", rparams.bgColor0);
		mBackgroundProgram.setVec3("uColor1", rparams.bgColor1);
		mScreenQuad.draw();

		glClear(GL_DEPTH_BUFFER_BIT);
		// draw a plane
		if (rparams.showGround)
		{
			mRenderHelper->drawGround(rparams.groudScale);
		}

		for (GLVisualModule* m : mRenderQueue)
		{
			// set material
			setMaterial(m); 
			m->paintGL(GLVisualModule::COLOR);
		}
		
		// draw scene bounding box
		if (rparams.showSceneBounds && scene != 0)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			glm::vec3 pmin = { p0[0], p0[1], p0[2] };
			glm::vec3 pmax = { p1[0], p1[1], p1[2] };
			mRenderHelper->drawBBox(pmin, pmax);
		}
		// draw axis
		if (rparams.showAxisHelper)
		{
			glViewport(10, 10, 100, 100);
			mRenderHelper->drawAxis();
		}
	}

	void RenderEngine::setMaterial(GLVisualModule* m)
	{
		struct MaterialProperty
		{
			glm::vec4	baseColor;
			float		metallic;
			float		roughness;
		} mtl;

		mtl.baseColor = glm::vec4(m->getColor(), m->getAlpha());
		mtl.metallic = m->getMetallic();
		mtl.roughness = m->getRoughness();

		//mMaterialUBO.load((void*)&mtl, sizeof(MaterialProperty));
	}

	void RenderEngine::updateShadowMap(const RenderParams& rparams)
	{
		mShadowMapUBO.bindBufferBase(0);
		mShadowMap->bind();

		for (GLVisualModule* m : mRenderQueue)
		{
			m->paintGL(GLVisualModule::DEPTH);
		}
	}		
}