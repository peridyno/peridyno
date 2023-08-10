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
#include <memory>

#include "screen.vert.h"
#include "blend.frag.h"

namespace dyno
{
	GLRenderEngine::GLRenderEngine()
	{

	}

	GLRenderEngine::~GLRenderEngine()
	{

	}

	void GLRenderEngine::initialize()
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

		createFramebuffer();

		// OIT
		setupTransparencyPass();

		gl::glCheckError();

		// create a screen quad
		mScreenQuad = gl::Mesh::ScreenQuad();

		mRenderHelper = new GLRenderHelper();
		mShadowMap = new ShadowMap(2048, 2048);
		mFXAAFilter = new FXAA;

	}

	void GLRenderEngine::terminate()
	{
		// release render modules
		for (auto item : mRenderItems) {
			item.visualModule->release();
		}

		// release framebuffer
		mFramebuffer.release();
		mColorTex.release();
		mDepthTex.release();
		mIndexTex.release();

		// release linked-list OIT objects
		mFreeNodeIdx.release();
		mLinkedListBuffer.release();
		mHeadIndexTex.release();
		mBlendProgram->release();
		delete mBlendProgram;

		// release other objects
		mScreenQuad->release();
		delete mScreenQuad;

		delete mRenderHelper;
		delete mShadowMap;
		delete mFXAAFilter;

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

		mBlendProgram = gl::Program::createProgramSPIRV(
			SCREEN_VERT, sizeof(SCREEN_VERT),
			BLEND_FRAG, sizeof(BLEND_FRAG));
	}


	void GLRenderEngine::createFramebuffer()
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

	void GLRenderEngine::updateRenderItems(dyno::SceneGraph* scene)
	{
		std::vector<RenderItem> items;
		for (auto iter = scene->begin(); iter != scene->end(); iter++) {
			for (auto m : iter->graphicsPipeline()->activeModules()) {
				if (auto vm = std::dynamic_pointer_cast<GLVisualModule>(m))
					items.push_back({ iter.get(), vm });
			}
		}

		// release GL resource for unreferenced visual module
		for (auto item : mRenderItems) {			
			if (std::find(items.begin(), items.end(), item) == items.end())
				item.visualModule->release();
		}
		mRenderItems = items;
	}

	void GLRenderEngine::draw(dyno::SceneGraph* scene, const RenderParams& rparams)
	{
		updateRenderItems(scene);

		// preserve current framebuffer
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// update framebuffer size
		resizeFramebuffer(rparams.width, rparams.height);

		// update shadow map
		mShadowMap->update(scene, rparams);

		// copy
		RenderParams params = rparams;
		// TODO: we might use world space
		params.light.mainLightDirection = glm::normalize(glm::vec3(
				params.transforms.view * glm::vec4(params.light.mainLightDirection, 0)));

		// bind internal framebuffer for rendering
		mFramebuffer.bind(GL_DRAW_FRAMEBUFFER);

		// attachement 0: color, attachment 1: index
		const unsigned int attachments[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		mFramebuffer.drawBuffers(2, attachments);

		// clear color and depth
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, rparams.width, rparams.height);

		// Step 1: draw background color, it also clears index buffer...
		{
			Vec3f c0 = Vec3f(this->bgColor0.x, this->bgColor0.y, this->bgColor0.z);
			Vec3f c1 = Vec3f(this->bgColor1.x, this->bgColor1.y, this->bgColor1.z);
			mRenderHelper->drawBackground(c0, c1);
		}

		mShadowMap->bind();

		// Step 2: render opacity objects
		{
			params.mode = GLRenderMode::COLOR;

			for (int i = 0; i < mRenderItems.size(); i++) 
			{
				if (mRenderItems[i].node->isVisible() && !mRenderItems[i].visualModule->isTransparent())
				{
					params.index = i;
					mRenderItems[i].visualModule->draw(params);
				}
			}
		}

		// Step 3: draw a ground grid (xy-plane)
		// since the grid is transparent, we handle it between opacity and transparent objects
		if (this->showGround)
		{
			// only draw to color buffer, so we can pick through
			mFramebuffer.drawBuffers(1, attachments);
			mRenderHelper->drawGround(params, this->planeScale, this->rulerScale);
		}

		// Step 4: transparency objects
		{
			// reset free node index
			const int zero = 0;
			mFreeNodeIdx.load((void*)&zero, sizeof(int));

			// reset head index
			const int clear = 0xFFFFFFFF;
			mHeadIndexTex.clear((void*)&clear);

			// binding...
			glBindImageTexture(0, mHeadIndexTex.id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
			mFreeNodeIdx.bindBufferBase(0);
			mLinkedListBuffer.bindBufferBase(0);

			// draw to no attachments
			mFramebuffer.drawBuffers(0, 0);

			// OIT: first pass
			glDepthMask(false);
			params.mode = GLRenderMode::TRANSPARENCY;
			for (int i = 0; i < mRenderItems.size(); i++) 
			{
				if (mRenderItems[i].node->isVisible() && mRenderItems[i].visualModule->isTransparent())
				{
					params.index = i;
					mRenderItems[i].visualModule->draw(params);
				}
			}
			glDepthMask(true);

			// OIT: blend alpha
			mFramebuffer.drawBuffers(2, attachments);
			mBlendProgram->use();
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
			mScreenQuad->draw();
			glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
		}		

		// Step 5: scene bounding box
		if (this->showSceneBounds && scene != 0)
		{
			mFramebuffer.drawBuffers(1, attachments);
			// get bounding box of the scene
			auto p0 = scene->getLowerBound();
			auto p1 = scene->getUpperBound();
			mRenderHelper->drawBBox(params, p0, p1);
		}

		// Step 6: draw to final framebuffer with fxaa filter
		{
			// restore previous framebuffer
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

			if (bEnableFXAA)
			{
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glViewport(0, 0, rparams.width, rparams.height);

				mColorTex.bind(GL_TEXTURE1);
				mDepthTex.bind(GL_TEXTURE2);
				mFXAAFilter->apply(rparams.width, rparams.height);
			}
			else
			{
				mFramebuffer.bind(GL_READ_FRAMEBUFFER);
				glReadBuffer(GL_COLOR_ATTACHMENT0);
				glBlitFramebuffer(
					0, 0, rparams.width, rparams.height,
					0, 0, rparams.width, rparams.height,
					GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
			}
		}

		gl::glCheckError();
	}


	void GLRenderEngine::resizeFramebuffer(int w, int h)
	{
		// resize internal framebuffer
		mColorTex.resize(w, h);
		mDepthTex.resize(w, h);
		mIndexTex.resize(w, h);
		mHeadIndexTex.resize(w, h);
		gl::glCheckError();
	}

	std::string GLRenderEngine::name() const
	{
		return std::string("Native OpenGL");
	}

	Selection GLRenderEngine::select(int x, int y, int w, int h)
	{
		// TODO: check valid input
		w = std::max(1, w);
		h = std::max(1, h);

		// read pixels
		std::vector<glm::ivec4> indices(w * h);

		mFramebuffer.bind(GL_READ_FRAMEBUFFER);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		//glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(x, y, w, h, GL_RGBA_INTEGER, GL_INT, indices.data());
		gl::glCheckError();

		// use unordered set to get unique id
		std::unordered_set<glm::ivec4> uniqueIdx(indices.begin(), indices.end());

		Selection result;
		result.x = x;
		result.y = y;
		result.w = w;
		result.h = h;

		for (const auto& idx : uniqueIdx) {
			const int nodeIdx = idx.x;
			const int instIdx = idx.y;
			const int primIdx = idx.z;

			if (nodeIdx >= 0 && nodeIdx < mRenderItems.size()) {
				result.items.push_back({mRenderItems[nodeIdx].node,	instIdx, primIdx});
			}
		}

		return result;
	}

}