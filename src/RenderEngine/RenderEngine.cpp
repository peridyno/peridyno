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
#include "module/FluidRender.h"
//#include "module/HeightFieldRender.h"

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
					m->display();
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
		mShadowMap = new ShadowMap();
	}

	RenderEngine::~RenderEngine()
	{

	}

	void RenderEngine::initialize()
	{
		if (!gladLoadGL()) {
			//SPDLOG_CRITICAL("Failed to load GLAD!");
			exit(-1);
		}
		else
		{
			//SPDLOG_INFO("Initialize RenderEngine with OpenGL {}.{}", GLVersion.major, GLVersion.minor);
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
		mSurfaceProgram = CreateShaderProgram("surface.vert", "surface.frag", "surface.geom");
		mPointProgram = CreateShaderProgram("point.vert", "point.frag");

		mPBRShadingProgram = CreateShaderProgram("screen.vert", "pbr.frag");
		mSSAOProgram = CreateShaderProgram("screen.vert", "ssao.frag");
		mFXAAProgram = CreateShaderProgram("screen.vert", "fxaa.frag");
		mBlendProgram = CreateShaderProgram("screen.vert", "blend.frag");

		// screen space fluid rendering
		mFluidProgram = CreateShaderProgram("fluid.vert", "fluid.frag");
		mFluidFilterProgram = CreateShaderProgram("screen.vert", "fluid.filter.frag");
		mFluidBlendProgram = CreateShaderProgram("screen.vert", "fluid.final.frag");
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

		// create uniform block for material
		mMaterialUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		mMaterialUBO.bindBufferBase(2);

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
		// basic camera transform...
		sceneUniformBuffer.view = glm::lookAt(
			rparams.camera.eye,
			rparams.camera.target,
			rparams.camera.up);
		// set projection matrix
		sceneUniformBuffer.projection = glm::perspective(
			rparams.camera.y_fov,
			rparams.camera.aspect,
			rparams.camera.z_min,
			rparams.camera.z_max);
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
		shadowUniformBuffer.width = 1024;
		shadowUniformBuffer.height = 1024;

		mShadowMapUBO.load(&shadowUniformBuffer, sizeof(shadowUniformBuffer));

		// light properties, convert into camera space
		RenderParams::Light light = rparams.light;
		light.mainLightDirection = glm::vec3(sceneUniformBuffer.view * glm::vec4(light.mainLightDirection, 0));
		light.mainLightVP = shadowUniformBuffer.projection * shadowUniformBuffer.view * glm::inverse(sceneUniformBuffer.view);
		mLightUBO.load(&light, sizeof(light));
	}


	void RenderEngine::draw(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams)
	{
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// pre-rendering 
		renderSetup(scene, target, rparams);
		updateShadowMap(rparams);

		// render background
		renderBackground(target, rparams);
		// render opacity objects
		renderOpacity(target, rparams);
		// render transparency objects
		renderTransparency(target, rparams);
		// fluid...
		renderFluid(target, rparams);
		// post processing
		renderPostprocess(target, rparams);

		target->drawColorTex();
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

			int			colorMode;
			float		colorMin;
			float		colorMax;

			int			shadowMode;
		} mtl;

		mtl.baseColor = glm::vec4(m->mBaseColor, m->mAlpha);
		mtl.metallic = m->mMetallic;
		mtl.roughness = m->mRoughness;
		mtl.colorMode = m->mColorMode;
		mtl.colorMin = m->mColorMin;
		mtl.colorMax = m->mColorMax;
		mtl.shadowMode = m->mShadowMode;

		mMaterialUBO.load((void*)&mtl, sizeof(MaterialProperty));
	}

	void RenderEngine::renderModule(GLVisualModule* m, unsigned int subroutine)
	{
 		if (auto* p = dynamic_cast<SurfaceRenderer*>(m))
 		{
 			mSurfaceProgram.use();
 			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
 			m->paintGL();
 		}
 		else if (auto* p = dynamic_cast<PointRenderer*>(m))
 		{
 			mPointProgram.use();
 			mPointProgram.setFloat("uPointSize", p->getPointSize());
 			glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
 			m->paintGL();
 		}
 		//else if (auto* p = dynamic_cast<HeightFieldRender*>(m))
 		//{
 		//	mSurfaceProgram.use();
 		//	glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
 		//	m->paintGL();
 		//}
 		else if (auto* p = dynamic_cast<FluidRenderer*>(m))
 		{
 			// skip...
 		}
 		else
 		{
 			//SPDLOG_ERROR("Unimplemented render module!");
 		}
	}

	void RenderEngine::updateShadowMap(const RenderParams& rparams)
	{
		// subroutine 1 for render shadow map
		const unsigned int subroutine = 1;

		mShadowMapUBO.bindBufferBase(0);
		mShadowMap->bind();

		for (GLVisualModule* m : mRenderQueue)
		{
			if (m->mShadowMode & GLVisualModule::ShadowMode::CAST)
			{
				if (!m->isTransparent())
				{
					renderModule(m, subroutine);
				}
			}
		}

		// transparent shadow map
		glClearColor(1.f, 1.f, 1.f, 0.f);
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_ZERO, GL_SRC_COLOR);
		glDepthMask(false);

		for (GLVisualModule* m : mRenderQueue)
		{
			if (m->mShadowMode & GLVisualModule::ShadowMode::CAST)
			{
				if (m->isTransparent())
				{
					setMaterial(m);
					renderModule(m, subroutine);
				}
			}
		}

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDepthMask(true);
		glDisable(GL_BLEND);
	}


	void RenderEngine::renderBackground(RenderTarget* target, const RenderParams& rparams)
	{
		mShadowMap->bindShadowTex();
		target->drawColorTex();
		glClearColor(rparams.bgColor.r, rparams.bgColor.g, rparams.bgColor.b, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw a plane
		if (rparams.showGround)
		{
			mTransformUBO.bindBufferBase(0);
			mRenderHelper->drawGround(rparams.groudScale);
		}
	}

	void RenderEngine::renderOpacity(RenderTarget* target, const RenderParams& rparams)
	{
		// 1. Generate G-buffer
		// since we use deferred shading...
		// use subroutine 0 for render G-buffer
		unsigned int subroutine = 0;

		mTransformUBO.bindBufferBase(0);
		target->drawGBufferTex();

		// render opacity objects
		glEnable(GL_DEPTH_TEST);
		for (GLVisualModule* m : mRenderQueue)
		{
			if (!m->isTransparent())
			{
				// set material
				setMaterial(m);
				renderModule(m, subroutine);
			}
		}

		// 2. Update SSAO texture
		// since SSAO is generated in screen space, we disable depth
		glDisable(GL_DEPTH_TEST);
		target->drawSSAOTex();
		// bind GBuffer textures
		target->bindGBufferTex();
		// bind SSAO noise texture
		mSSAONoiseTex.bind(GL_TEXTURE4);
		mSSAOProgram.use();
		mScreenQuad.draw();

		// 3. Deferred shading in screen space
		target->drawColorTex();
		mShadowMap->bindShadowTex();
		target->bindGBufferTex();
		target->bindSSAOTex();

		mPBRShadingProgram.use();
		mScreenQuad.draw();
		glEnable(GL_DEPTH_TEST);

	}

	void RenderEngine::renderFluid(RenderTarget* target, const RenderParams& rparams)
	{
		int idx = 0;

		// render depth and thick
		mFluidProgram.use();
		mFluidProgram.setFloat("uScreenWidth", target->width);

		target->drawFluidTex(idx);

		// clear texture
		glClearColor(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f);
		glClear(GL_COLOR_BUFFER_BIT);

		// disable depth write
		glEnable(GL_DEPTH_TEST);
		glDepthMask(false);

		// blend function
		glEnable(GL_BLEND);
		glBlendEquationSeparate(GL_MAX, GL_FUNC_ADD);
		glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);

		for (GLVisualModule* m : mRenderQueue)
		{
			if (auto* p = dynamic_cast<FluidRenderer*>(m))
			{
				mFluidProgram.setFloat("uPointRadius", p->getPointSize());
				m->paintGL();
			}
		}
		glDepthMask(true);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDisable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);


		// filter depth and thickness
		mFluidFilterProgram.use();
		mFluidFilterProgram.setInt("viewportWidth", target->width);
		mFluidFilterProgram.setInt("viewportHeight", target->height);

		for (int iter = 0; iter < 3; iter++)
		{
			idx = 1 - idx;
			target->drawFluidTex(idx);
			target->bindFluidTex(GL_TEXTURE0, 1 - idx);
			mScreenQuad.draw();
		}

		// render fluid
		mFluidBlendProgram.use();
		mFluidBlendProgram.setInt("viewportWidth", target->width);
		mFluidBlendProgram.setInt("viewportHeight", target->height);

		GLuint subroutine = 0;
		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);

		idx = 1 - idx;
		target->drawFluidTex(idx);
		target->bindFluidTex(GL_TEXTURE0, 1 - idx);
		target->bindColorTex();
		mScreenQuad.draw();

		// final blend
		subroutine = 1;
		glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &subroutine);
		target->bindFluidTex(GL_TEXTURE0, idx);
		target->drawColorTex();
		mScreenQuad.draw();

	}

	void RenderEngine::renderTransparency(RenderTarget* target, const RenderParams& rparams)
	{
		// use subrouting for linked-list
		const unsigned int subroutine = 2;

		mTransformUBO.bindBufferBase(0);

		// render linked list
		// note: we should enable early-z and disable depth write
		target->drawOITLinkedList();
		mShadowMap->bindShadowTex();

		glDepthMask(false);
		for (GLVisualModule* m : mRenderQueue)
		{
			if (m->isTransparent())
			{
				// set material
				setMaterial(m);
				renderModule(m, subroutine);
			}
		}
		glDepthMask(true);

		// update depth buffer
		for (GLVisualModule* m : mRenderQueue)
		{
			if (m->isTransparent())
			{
				renderModule(m, 1);
			}
		}

		// blend transparency
		target->drawColorTex();
		mBlendProgram.use();

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);

		mScreenQuad.draw();

		glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);

	}

	void RenderEngine::renderPostprocess(RenderTarget* target, const RenderParams& rparams)
	{
		return;
		//glViewport(rparams.viewport.x, rparams.viewport.y, rparams.viewport.w, rparams.viewport.h);
		//glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		//glClearColor(rparams.bgColor.r, rparams.bgColor.g, rparams.bgColor.b, 1.f);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//// for test only
		//if (true)
		//{
		//	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		//	glBindFramebuffer(GL_READ_FRAMEBUFFER, target->mFramebuffer.id);
		//	glReadBuffer(GL_COLOR_ATTACHMENT0);
		//	glBlitFramebuffer(
		//		rparams.viewport.x, rparams.viewport.y, rparams.viewport.w, rparams.viewport.h,
		//		rparams.viewport.x, rparams.viewport.y, rparams.viewport.w, rparams.viewport.h, 
		//		GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		//	return;
		//}

		// FXAA
		const GLfloat g_lumaThreshold = 0.5f;
		const GLfloat g_mulReduceReciprocal = 8.0f;
		const GLfloat g_minReduceReciprocal = 128.0f;
		const GLfloat g_maxSpan = 4.0f;

		mFXAAProgram.use();
		mFXAAProgram.setVec2("u_texelStep", glm::vec2(1.f / rparams.viewport.w, 1.f / rparams.viewport.h));
		mFXAAProgram.setFloat("u_lumaThreshold", g_lumaThreshold);
		mFXAAProgram.setFloat("u_mulReduce", 1.0f / g_mulReduceReciprocal);
		mFXAAProgram.setFloat("u_minReduce", 1.0f / g_minReduceReciprocal);
		mFXAAProgram.setFloat("u_maxSpan", g_maxSpan);

		target->bindColorTex();

		glEnable(GL_BLEND);
		mScreenQuad.draw();
		glDisable(GL_BLEND);

	}

}