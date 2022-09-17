#include "GLRenderEngine.h"
#include "GLRenderHelper.h"
#include "GLVisualModule.h"

#include "Utility.h"
#include "ShadowMap.h"
#include "SSAO.h"
#include "FXAA.h"

// dyno
#include "SceneGraph.h"
#include "Action.h"

// GLM
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/glad.h>

#include <OrbitCamera.h>
#include <TrackballCamera.h>

namespace dyno
{
	class RenderQueue : public Action
	{
	public:
		RenderQueue() {};
		~RenderQueue() override { modules.clear(); }

		void draw(GLRenderPass pass)
		{
			for (GLVisualModule* m : modules)
				m->draw(pass);
		}

	private:
		void process(Node* node) override
		{
			if (!node->isVisible())
				return;

			for (auto iter : node->graphicsPipeline()->activeModules())
			{
				auto m = dynamic_cast<GLVisualModule*>(iter.get());
				if (m && m->isVisible())
				{
					//m->update();
					modules.push_back(m);
				}
			}
		}
		std::vector<GLVisualModule*> modules;
	};

	GLRenderEngine::GLRenderEngine()
	{
		mRenderHelper = new GLRenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
		mSSAO = new SSAO;
		mFXAAFilter = new FXAA;

		setupCamera();
	}

	GLRenderEngine::~GLRenderEngine()
	{
		delete mRenderHelper;
		delete mShadowMap;
		delete mSSAO;
		delete mFXAAFilter;
	}

	void GLRenderEngine::initialize(int width, int height)
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

		setupInternalFramebuffer();

		// create uniform block for transform
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		// create uniform block for light
		mLightUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		gl::glCheckError();

		mSSAO->initialize();
		mShadowMap->initialize();
		mRenderHelper->initialize();
		mFXAAFilter->initialize();

		mCamera->setEyePos(Vec3f(1.5f, 1.0f, 1.5f));

		this->resize(width, height);
	}

	void GLRenderEngine::setupCamera()
	{
		switch (mCameraType)
		{
		case dyno::Orbit:
			mCamera = std::make_shared<OrbitCamera>();
			break;
		case dyno::TrackBall:
			mCamera = std::make_shared<TrackballCamera>();
			break;
		default:
			break;
		}
	}

	void GLRenderEngine::setupInternalFramebuffer()
	{
		// create textures
		mColorTex.maxFilter = GL_LINEAR;
		mColorTex.minFilter = GL_LINEAR;
		mColorTex.create();
		mColorTex.resize(1, 1);

		mDepthTex.internalFormat = GL_DEPTH_COMPONENT32;
		mDepthTex.format = GL_DEPTH_COMPONENT;
		mDepthTex.create();
		mDepthTex.resize(1, 1);

		// index
		mIndexTex.internalFormat = GL_RGBA32I;
		mIndexTex.format = GL_RGBA_INTEGER;
		mIndexTex.type = GL_INT;
		mIndexTex.create();
		mIndexTex.resize(1, 1);

		// create framebuffer
		mFramebuffer.create();

		// bind framebuffer texture
		mFramebuffer.bind();
		mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mDepthTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT0, mColorTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT1, mIndexTex.id);

		const GLenum buffers[] = {
			GL_COLOR_ATTACHMENT0,
			GL_COLOR_ATTACHMENT1
		};
		mFramebuffer.drawBuffers(2, buffers);

		mFramebuffer.checkStatus();
		mFramebuffer.unbind();
		gl::glCheckError();
	}

	void GLRenderEngine::draw(dyno::SceneGraph* scene)
	{
		// preserve current framebuffer
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// update render parameters
		m_rparams.proj = mCamera->getProjMat();
		m_rparams.view = mCamera->getViewMat();

		// update shadow map
		mShadowMap->update(scene, m_rparams);

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
		sceneUniformBuffer.view = m_rparams.view;
		sceneUniformBuffer.projection = m_rparams.proj;
		sceneUniformBuffer.width = m_rparams.viewport.w;
		sceneUniformBuffer.height = m_rparams.viewport.h;

		mTransformUBO.load(&sceneUniformBuffer, sizeof(sceneUniformBuffer));
		mTransformUBO.bindBufferBase(0);

		// setup light block
		RenderParams::Light light = m_rparams.light;
		light.mainLightDirection = glm::vec3(m_rparams.view * glm::vec4(light.mainLightDirection, 0));
		mLightUBO.load(&light, sizeof(light));
		mLightUBO.bindBufferBase(1);

		
		mFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
		glViewport(0, 0, m_rparams.viewport.w, m_rparams.viewport.h);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// draw background color
		Vec3f c0 = Vec3f(m_rparams.bgColor0.x, m_rparams.bgColor0.y, m_rparams.bgColor0.z);
		Vec3f c1 = Vec3f(m_rparams.bgColor1.x, m_rparams.bgColor1.y, m_rparams.bgColor1.z);
		mRenderHelper->drawBackground(c0, c1);

		// draw a ruler plane, since it's opacity
		if (m_rparams.showGround)
		{
			mRenderHelper->drawGround(
				m_rparams.planeScale * mCamera->distanceUnit(),
				m_rparams.rulerScale * mCamera->distanceUnit());
		}

		// render visual modules
		RenderQueue renderQueue;
		// enqueue render content
		if (scene != nullptr && !scene->isEmpty()) {
			scene->traverseForward(&renderQueue);
		}
		renderQueue.draw(GLRenderPass::COLOR);
		
		// draw scene bounding box
		if (m_rparams.showSceneBounds && scene != 0)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			mRenderHelper->drawBBox(p0, p1);
		}
		// draw axis
		if (m_rparams.showAxisHelper)
		{
			glViewport(10, 10, 100, 100);
			mRenderHelper->drawAxis();
		}

		// draw to final framebuffer with fxaa filter
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, m_rparams.viewport.w, m_rparams.viewport.h);
		
		if (m_rparams.useFXAA)
		{
			mColorTex.bind(GL_TEXTURE1);
			mFXAAFilter->apply(m_rparams.viewport.w, m_rparams.viewport.h);
		}
		else
		{
			mFramebuffer.bind(GL_READ_FRAMEBUFFER);
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glBlitFramebuffer(
				0, 0, m_rparams.viewport.w, m_rparams.viewport.h,
				0, 0, m_rparams.viewport.w, m_rparams.viewport.h,
				GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		}

		gl::glCheckError();
	}

	void GLRenderEngine::resize(int w, int h)
	{
		// set the viewport
		m_rparams.viewport.x = 0;
		m_rparams.viewport.y = 0;
		m_rparams.viewport.w = w;
		m_rparams.viewport.h = h;

		mCamera->setWidth(w);
		mCamera->setHeight(h);

		// resize internal framebuffer
		mColorTex.resize(w, h);
		mDepthTex.resize(w, h);
		mIndexTex.resize(w, h);
		gl::glCheckError();
	}

	std::string GLRenderEngine::name()
	{
		return std::string("Native OpenGL");
	}

}