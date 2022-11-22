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
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <glad/glad.h>

#include <OrbitCamera.h>
#include <TrackballCamera.h>
#include <unordered_set>

namespace dyno
{
	GLRenderEngine::GLRenderEngine()
	{
		mRenderHelper = new GLRenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
		mSSAO = new SSAO;
		mFXAAFilter = new FXAA;
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
			printf("Failed to load OpenGL context!");
			exit(-1);
		}

		// some basic opengl settings
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);
		glDepthFunc(GL_LEQUAL);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		setupInternalFramebuffer();

		// OIT
		setupTransparencyPass();

		mScreenQuad = gl::Mesh::ScreenQuad();

		// create uniform block for transform
		mTransformUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		// create uniform block for light
		mLightUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		// create uniform bnlock for global variables
		mVariableUBO.create(GL_UNIFORM_BUFFER, GL_DYNAMIC_DRAW);
		gl::glCheckError();

		mSSAO->initialize();
		mShadowMap->initialize();
		mRenderHelper->initialize();
		mFXAAFilter->initialize();

		this->resize(width, height);

	}

	void GLRenderEngine::setupTransparencyPass()
	{
		mFreeNodeIdx.create(GL_ATOMIC_COUNTER_BUFFER, GL_DYNAMIC_DRAW);
		mFreeNodeIdx.allocate(sizeof(int));

		mLinkedListBuffer.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		struct NodeType
		{
			glm::vec4 color;
			float	  depth;
			unsigned int next;
			unsigned int idx0;
			unsigned int idx1;
		};
		mLinkedListBuffer.allocate(sizeof(NodeType) * MAX_OIT_NODES);

		// transparency
		mHeadIndexTex.internalFormat = GL_R32UI;
		mHeadIndexTex.format = GL_RED_INTEGER;
		mHeadIndexTex.type = GL_UNSIGNED_INT;
		mHeadIndexTex.create();
		mHeadIndexTex.resize(1, 1);

		mBlendProgram = gl::ShaderFactory::createShaderProgram("screen.vert", "blend.frag");
	}


	void GLRenderEngine::setupInternalFramebuffer()
	{
		// create render textures
		mColorTex.maxFilter = GL_LINEAR;
		mColorTex.minFilter = GL_LINEAR;
		mColorTex.format = GL_RGBA;
		mColorTex.internalFormat = GL_RGBA;
		mColorTex.type = GL_BYTE;
		mColorTex.create();
		mColorTex.resize(1, 1);

		mDepthTex.internalFormat = GL_DEPTH_COMPONENT32;
		mDepthTex.format = GL_DEPTH_COMPONENT;
		mDepthTex.create();
		mDepthTex.resize(1, 1);

		// index
		mIndexTex.internalFormat = GL_RGBA32I;
		mIndexTex.format = GL_RGBA_INTEGER;
		mIndexTex.type   = GL_INT;
		//mIndexTex.wrapS = GL_CLAMP_TO_EDGE;
		//mIndexTex.wrapT = GL_CLAMP_TO_EDGE;
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

	void GLRenderEngine::updateRenderModules(dyno::SceneGraph* scene)
	{
		// render visual modules
		struct RenderQueue : public Action {
			void process(Node* node) override
			{
				if (!node->isVisible())	return;
				for (auto iter : node->graphicsPipeline()->activeModules()) {
					auto m = dynamic_cast<GLVisualModule*>(iter.get());
					if (m && m->isVisible()) {
						modules.push_back(m);
						nodes.push_back(node);
					}
				}
			}
			std::vector<GLVisualModule*> modules;
			std::vector<Node*>			 nodes;
		} action;

		// enqueue modules for rendering
		if (scene != nullptr && !scene->isEmpty()) {
			scene->traverseForward(&action);
		}

		mRenderModules = action.modules;
		mRenderNodes   = action.nodes;
	}

	void GLRenderEngine::draw(dyno::SceneGraph* scene)
	{
		updateRenderModules(scene);

		// preserve current framebuffer
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

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

		// clear color and depth
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		// clear index buffer
		//GLint clearIndex[]{11, -1, -1, -1};
		//glClearBufferiv(GL_COLOR, 1, clearIndex);
		glViewport(0, 0, m_rparams.viewport.w, m_rparams.viewport.h);
		// draw background color
		Vec3f c0 = Vec3f(m_rparams.bgColor0.x, m_rparams.bgColor0.y, m_rparams.bgColor0.z);
		Vec3f c1 = Vec3f(m_rparams.bgColor1.x, m_rparams.bgColor1.y, m_rparams.bgColor1.z);
		mRenderHelper->drawBackground(c0, c1);
		
		mVariableUBO.bindBufferBase(2);

		// render opacity objects
		for (int i = 0; i < mRenderModules.size(); i++) {

			if (!mRenderModules[i]->isTransparent())
			{
				mVariableUBO.load(&i, sizeof(i));
				mRenderModules[i]->draw(GLRenderPass::COLOR);
			}
		}

		// render transparency objects
		{
			// reset free node index
			const int zero = 0;
			mFreeNodeIdx.load((void*)&zero, sizeof(int));
			// reset head index
			const int clear = 0xFFFFFFFF;
			mHeadIndexTex.clear((void*)&clear);

			glBindImageTexture(0, mHeadIndexTex.id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
			mFreeNodeIdx.bindBufferBase(0);
			mLinkedListBuffer.bindBufferBase(0);

			// draw to no attachments
			mFramebuffer.drawBuffers(0, 0);

			glDepthMask(false);
			for (int i = 0; i < mRenderModules.size(); i++) {

				if (mRenderModules[i]->isTransparent())
				{
					mVariableUBO.load(&i, sizeof(i));
					mRenderModules[i]->draw(GLRenderPass::TRANSPARENCY);
				}
			}

			// draw a ruler plane
			if (m_rparams.showGround)
			{
				mRenderHelper->drawGround(m_rparams.planeScale, m_rparams.rulerScale);
			}

			glDepthMask(true);

			// blend alpha
			const unsigned int attachments[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
			mFramebuffer.drawBuffers(2, attachments);

			mBlendProgram.use();

			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);

			mScreenQuad.draw();

			glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
		}


		// draw scene bounding box
		if (m_rparams.showSceneBounds && scene != 0)
		{
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			mRenderHelper->drawBBox(p0, p1);
		}

		// draw to final framebuffer with fxaa filter
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
				
		if (m_rparams.useFXAA)
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glViewport(0, 0, m_rparams.viewport.w, m_rparams.viewport.h);

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

		// draw axis
		if (m_rparams.showAxisHelper)
		{
			glViewport(10, 10, 100, 100);
			mRenderHelper->drawAxis();
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

		// resize internal framebuffer
		mColorTex.resize(w, h);
		mDepthTex.resize(w, h);
		mIndexTex.resize(w, h);

		// transparency
		mHeadIndexTex.resize(w, h);

		gl::glCheckError();
	}

	std::string GLRenderEngine::name()
	{
		return std::string("Native OpenGL");
	}

	std::vector<SelectionItem> GLRenderEngine::select(int x, int y, int w, int h)
	{
		// TODO: check valid input
		w = std::max(1, w);
		h = std::max(1, h);

		// read pixels
		std::vector<glm::ivec4> indices(w * h);

		mFramebuffer.bind(GL_READ_FRAMEBUFFER);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		//glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(x, m_rparams.viewport.h - (y + h) - 1, w, h, GL_RGBA_INTEGER, GL_INT, indices.data());
		gl::glCheckError();

		// use unordered set to get unique id
		std::unordered_set<glm::ivec4> uniqueIdx(indices.begin(), indices.end());

		std::vector<SelectionItem> items;

		for (glm::ivec4 i : uniqueIdx) {
			int nodeIdx = i.x;

			if (nodeIdx >= 0 && nodeIdx < mRenderNodes.size()) {
				SelectionItem item;
				item.node = mRenderNodes[nodeIdx];
				item.instance = i.y;

				items.push_back(item);
			}
		}

		return items;
	}

}