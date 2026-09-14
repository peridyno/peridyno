#include "GLRenderEngine.h"
#include "GLRenderHelper.h"
#include "GLVisualModule.h"

#include "Utility.h"
#include "ShadowMap.h"
#include "SSAO.h"
#include "FXAA.h"
#include "Envmap.h"

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
#include "wboit_composite.frag.h"
#include "postprocess.frag.h"
#include "surface.frag.h"

namespace dyno
{
	GLRenderEngine::GLRenderEngine()
	{
		mShadowMap = new ShadowMap();
		mEnvmap = new Envmap();
	}

	GLRenderEngine::~GLRenderEngine()
	{
		delete mShadowMap;
		delete mEnvmap;
		delete mScreenQuad;
		delete mWBOITCompositeProgram;
		delete mRenderHelper;
		delete mFXAAFilter;
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

		glCheckError();

		// create a screen quad
		mScreenQuad = Mesh::ScreenQuad();

		mRenderHelper = new GLRenderHelper();
		mFXAAFilter = new FXAA;

		mShadowMap->initialize();
		mEnvmap->initialize();

		this->setDefaultEnvmap();
	}

	void GLRenderEngine::terminate()
	{
		mShadowMap->release();
		mEnvmap->release();

		// release render modules
		for (auto item : mRenderItems) {
			item.visualModule->release();
		}

		// release framebuffer
		mFramebuffer.release();
		mColorTex.release();
		mDepthTex.release();
		mIndexTex.release();

		mSelectIndexTex.release();
		mSelectFramebuffer.release();

		// release WBOIT objects
		mAccumTex.release();
		mRevealTex.release();
		mAccumResolveTex.release();
		mRevealResolveTex.release();
		mWBOITFramebuffer.release();
		mWBOITResolveFBO.release();
		mWBOITCompositeProgram->release();

		// release other objects
		mScreenQuad->release();

	}

	void GLRenderEngine::setupTransparencyPass()
	{
		// Weighted Blended OIT G-buffers (RGBA16F, multisample to match MSAA)
		mAccumTex.internalFormat = GL_RGBA16F;
		mAccumTex.format = GL_RGBA;
		mAccumTex.type = GL_HALF_FLOAT;
		mAccumTex.create();
		mAccumTex.resize(1, 1, 1);

		mRevealTex.internalFormat = GL_RGBA16F;
		mRevealTex.format = GL_RGBA;
		mRevealTex.type = GL_HALF_FLOAT;
		mRevealTex.create();
		mRevealTex.resize(1, 1, 1);

		// single-sample resolve targets for the composite pass
		mAccumResolveTex.internalFormat = GL_RGBA16F;
		mAccumResolveTex.format = GL_RGBA;
		mAccumResolveTex.type = GL_HALF_FLOAT;
		mAccumResolveTex.create();
		mAccumResolveTex.resize(1, 1);

		mRevealResolveTex.internalFormat = GL_RGBA16F;
		mRevealResolveTex.format = GL_RGBA;
		mRevealResolveTex.type = GL_HALF_FLOAT;
		mRevealResolveTex.create();
		mRevealResolveTex.resize(1, 1);

		// WBOIT framebuffer: accum(0) + reveal(2) + the opaque depth (shared)
		mWBOITFramebuffer.create();
		mWBOITFramebuffer.bind();
		mWBOITFramebuffer.setTexture(GL_DEPTH_ATTACHMENT, &mDepthTex);
		mWBOITFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mAccumTex);
		mWBOITFramebuffer.setTexture(GL_COLOR_ATTACHMENT2, &mRevealTex);
		// loc0=accum(COLOR_ATTACHMENT0), loc1=fragIndices(GL_NONE, unused here),
		// loc2=revealage(COLOR_ATTACHMENT2). draw-buffer index == shader location.
		const GLenum wboitBuffers[] = { GL_COLOR_ATTACHMENT0, GL_NONE, GL_COLOR_ATTACHMENT2 };
		mWBOITFramebuffer.drawBuffers(3, wboitBuffers);
		mWBOITFramebuffer.checkStatus();
		mWBOITFramebuffer.unbind();

		// single-sample resolve FBO for the composite pass
		mWBOITResolveFBO.create();
		mWBOITResolveFBO.bind();
		mWBOITResolveFBO.setTexture(GL_COLOR_ATTACHMENT0, &mAccumResolveTex);
		mWBOITResolveFBO.setTexture(GL_COLOR_ATTACHMENT2, &mRevealResolveTex);
		mWBOITResolveFBO.drawBuffers(2, wboitBuffers);
		mWBOITResolveFBO.checkStatus();
		mWBOITResolveFBO.unbind();

		// composite pass: blend resolved transparent layer over opaque color
		mWBOITCompositeProgram = Program::createProgramSPIRV(
			SCREEN_VERT, sizeof(SCREEN_VERT),
			WBOIT_COMPOSITE_FRAG, sizeof(WBOIT_COMPOSITE_FRAG));

		mPostProcessProgram = Program::createProgramSPIRV(
			SCREEN_VERT, sizeof(SCREEN_VERT),
			POSTPROCESS_FRAG, sizeof(POSTPROCESS_FRAG));
	}


	void GLRenderEngine::setShadowMapSize(int size)
	{
		mShadowMap->setSize(size);
	}

	int GLRenderEngine::getShadowMapSize() const
	{
		return mShadowMap->getSize();
	}

	void GLRenderEngine::setShadowBlurIters(int iters)
	{
		mShadowMap->setNumBlurIterations(iters);
	}

	int GLRenderEngine::getShadowBlurIters() const
	{
		return mShadowMap->getNumBlurIterations();
	}

	void GLRenderEngine::setDefaultEnvmap()
	{
		setEnvmap(getAssetPath() + "textures/hdr/venice_dawn_1_4k.hdr");
	}

	void GLRenderEngine::setEnvmap(const std::string& file)
	{
		if (file.empty()) {
			//bDrawEnvmap = false;
			return;
		}
		else
		{
			//bDrawEnvmap = true;
			mEnvmapFilePath = file;
			mEnvmap->load(file.c_str());
		}
	}

	void GLRenderEngine::setEnvStyle(EEnvStyle style)
	{
		envStyle = style;

		if (style == EEnvStyle::Standard)
		{
			this->bgColor0 = glm::vec3(0.2f);
			this->bgColor1 = glm::vec3(0.8f);

			this->planeColor = { 0.3, 0.3, 0.3, 0.5 };
			this->rulerColor = { 0.0, 0.0, 0.0, 0.5 };

			this->setUseEnvmapBackground(false);
			this->setEnvmapScale(0.0f);
		}
		else if (style == EEnvStyle::Studio)
		{
			this->bgColor0 = { 1, 1, 1 };
			this->bgColor1 = { 1, 1, 1 };

			this->planeColor = { 1,1,1,1 };
			this->rulerColor = { 1,1,1,1 };

			this->setUseEnvmapBackground(true);
			this->setEnvmapScale(1.0f);
		}
	}

	int GLRenderEngine::getShadowMapSize()
	{
		if (!mShadowMap)
		{
			shadowQuality = mShadowMap->getSize();
		}
		return shadowQuality;
	}

	void GLRenderEngine::updateShadowMapAttribute()
	{
		if (mShadowMap) 
		{
			mShadowMap->setSize(shadowQuality);
			mShadowMap->clampToSceneBounds = bClampToSceneBound;
			mShadowMap->useSceneBounds = bUseSceneBoundForShadow;
		}
	}

	void GLRenderEngine::createFramebuffer()
	{
		// create render textures
		mColorTex.format = GL_RGBA;
		mColorTex.internalFormat = GL_RGBA;
		mColorTex.type = GL_BYTE;
		mColorTex.create();
		mColorTex.resize(1, 1, 1);

		mDepthTex.internalFormat = GL_DEPTH_COMPONENT32;
		mDepthTex.format = GL_DEPTH_COMPONENT;
		mDepthTex.create();
		mDepthTex.resize(1, 1, 1);

		// index
		mIndexTex.internalFormat = GL_RGBA32I;
		mIndexTex.format = GL_RGBA_INTEGER;
		mIndexTex.type   = GL_INT;
		//mIndexTex.wrapS = GL_CLAMP_TO_EDGE;
		//mIndexTex.wrapT = GL_CLAMP_TO_EDGE;
		mIndexTex.create();
		mIndexTex.resize(1, 1, 1);

		// create framebuffer
		mFramebuffer.create();

		// bind framebuffer texture
		mFramebuffer.bind();
		mFramebuffer.setTexture(GL_DEPTH_ATTACHMENT, &mDepthTex);
		mFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mColorTex);
		mFramebuffer.setTexture(GL_COLOR_ATTACHMENT1, &mIndexTex);

		const GLenum buffers[] = {
			GL_COLOR_ATTACHMENT0,
			GL_COLOR_ATTACHMENT1
		};
		mFramebuffer.drawBuffers(2, buffers);

		mFramebuffer.checkStatus();
		mFramebuffer.unbind();

		// select framebuffer
		mSelectIndexTex.internalFormat = GL_RGBA32I;
		mSelectIndexTex.format = GL_RGBA_INTEGER;
		mSelectIndexTex.type = GL_INT;
		mSelectIndexTex.create();
		mSelectIndexTex.resize(1, 1);

		mSelectFramebuffer.create();
		mSelectFramebuffer.bind();
		mSelectFramebuffer.setTexture(GL_COLOR_ATTACHMENT0, &mSelectIndexTex);
		mSelectFramebuffer.drawBuffers(1, buffers);
		mSelectFramebuffer.checkStatus();
		mSelectFramebuffer.unbind();

		glCheckError();
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

	void GLRenderEngine::draw(dyno::SceneGraph* scene, const RenderParams& rparams, Vec2i p)
	{
		updateRenderItems(scene);

		// preserve current framebuffer
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// resize internal framebuffer
		GLint samples;
		glGetFramebufferParameteriv(GL_FRAMEBUFFER, GL_SAMPLES, &samples);
		if (bEnableFXAA) {
			// if FXAA is enabled, we use 1 spp internal framebuffer
			resizeFramebuffer(rparams.width, rparams.height, 1);
		}
		else if (samples > 0) {
			// external framebuffer MSAA is enabled,
			resizeFramebuffer(rparams.width, rparams.height, samples);
		}
		else {
			// target framebuffer is non-multisample, and FXAA is disabled...
			resizeFramebuffer(rparams.width, rparams.height, mMSAASamples);
		}

		// update shadow map
		mShadowMap->update(scene, rparams);

		// copy
		RenderParams params = rparams;
		// TODO: we might use world space
		params.light.mainLightDirection = glm::normalize(glm::vec3(
				params.transforms.view * glm::vec4(params.light.mainLightDirection, 0)));

		// Helper: set the main-light shadow flag on params according to whether
		// the module wants to receive shadows. Returns the previous value so the
		// caller can restore it (params is reused across draw calls).
		auto applyReceiveShadow = [&params](const std::shared_ptr<GLVisualModule>& vm) -> float
		{
			float prev = params.light.mainLightShadow;
			if (!vm->varReceiveShadow()->getValue())
				params.light.mainLightShadow = 0.f;
			return prev;
		};

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
			mRenderHelper->drawBackground(
				Vec3f(this->bgColor0.x, this->bgColor0.y, this->bgColor0.z), 
				Vec3f(this->bgColor1.x, this->bgColor1.y, this->bgColor1.z));
		}

		//
		if(bDrawEnvmap) {
			mEnvmap->draw(params);
		}

		// clear index buffer
		GLint clearIndex[] = { -1, -1, -1, -1 };
		glClearBufferiv(GL_COLOR, 1, clearIndex);
		glCheckError();

		mShadowMap->bind();

		mEnvmap->setScale(enmapScale);
		mEnvmap->bindIBL();

		// Step 2: render opacity objects
		{
			params.mode = GLRenderMode::COLOR;

			for (int i = 0; i < mRenderItems.size(); i++) 
			{
				if (mRenderItems[i].node->isVisible()
					&& !mRenderItems[i].visualModule->isTransparent()
					&& !mRenderItems[i].visualModule->varRenderToFront()->getValue())
				{
					params.index = i;
					float savedShadow = applyReceiveShadow(mRenderItems[i].visualModule);
					mRenderItems[i].visualModule->draw(params);
					params.light.mainLightShadow = savedShadow;
				}
			}
		}

		// Step 3: draw a ground grid (xy-plane)
		// since the grid is transparent, we handle it between opacity and transparent objects
		if (this->showGround)
		{
			float unitScale = rparams.unitScale;
			// only draw to color buffer, so we can pick through
			mFramebuffer.drawBuffers(1, attachments);
			mRenderHelper->drawGround(params,
				this->planeScale * unitScale, this->rulerScale * unitScale,
				Vec4f(this->planeColor.r, this->planeColor.g, this->planeColor.b, this->planeColor.a),
				Vec4f(this->rulerColor.r, this->rulerColor.g, this->rulerColor.b, this->rulerColor.a));
		}

		// Step 4: transparency objects (Weighted Blended OIT)
		{
			params.mode = GLRenderMode::TRANSPARENCY;

			// bind the WBOIT framebuffer, depth-tested against the opaque depth
		mWBOITFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
		// Map fragment-shader output locations to attachments:
		//   loc0 = fragColor  (accum)      -> draw-buffer index 0 -> COLOR_ATTACHMENT0
		//   loc1 = fragIndices (picking, unused here) -> index 1 -> GL_NONE (discarded)
		//   loc2 = outReveal  (revealage)  -> draw-buffer index 2 -> COLOR_ATTACHMENT2
		// gl_FragData[location] routes to draw-buffer INDEX == location, so the
		// revealage output MUST sit at index 2 (matching its shader location).
		const unsigned int wboitBuffers[] = { GL_COLOR_ATTACHMENT0, GL_NONE, GL_COLOR_ATTACHMENT2 };
		mWBOITFramebuffer.drawBuffers(3, wboitBuffers);

		// clear accum -> (0,0,0,0), revealage -> (1,1,1,1) (fully transparent)
		// glClearBufferfv's 2nd arg is the DRAW-BUFFER INDEX (0-based), NOT the
		// GL_COLOR_ATTACHMENTn enum. accum is index 0, revealage is index 2.
		const float accumClear[4] = { 0.f, 0.f, 0.f, 0.f };
		const float revealClear[4] = { 1.f, 1.f, 1.f, 1.f };
		glClearBufferfv(GL_COLOR, 0, accumClear);
		glClearBufferfv(GL_COLOR, 2, revealClear);

		// depth test against opaque (read-only, no depth writes).
		// Use GL_LESS (strictly closer) instead of GL_LEQUAL: a coplanar /
		// coincident transparent surface then cleanly FAILS the test (hidden
		// behind the opaque one) instead of fighting frame-to-frame, which
		// would otherwise manifest as flickering at opaque/transparent overlap.
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_FALSE);

		// per-attachment blending (matched to the draw-buffer indices above):
		//   accum   (idx 0): additive          (ONE, ONE)
		//   revealage(idx 2): multiplicative    (ZERO, ONE_MINUS_SRC_ALPHA)
		glEnable(GL_BLEND);
		glBlendFunci(0, GL_ONE, GL_ONE);
		glBlendFunci(2, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

			for (int i = 0; i < mRenderItems.size(); i++)
			{
				if (mRenderItems[i].node->isVisible()
					&& mRenderItems[i].visualModule->isTransparent()
					&& !mRenderItems[i].visualModule->varRenderToFront()->getValue())
				{
					params.index = i;
					float savedShadow = applyReceiveShadow(mRenderItems[i].visualModule);
					mRenderItems[i].visualModule->draw(params);
					params.light.mainLightShadow = savedShadow;
				}
			}

			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);

			// resolve the (possibly multisample) G-buffers to single-sample targets
			mWBOITFramebuffer.bind(GL_READ_FRAMEBUFFER);
			mWBOITResolveFBO.bind(GL_DRAW_FRAMEBUFFER);
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);
			glBlitFramebuffer(0, 0, rparams.width, rparams.height,
				0, 0, rparams.width, rparams.height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
		glReadBuffer(GL_COLOR_ATTACHMENT2);
		glDrawBuffer(GL_COLOR_ATTACHMENT2);
			glBlitFramebuffer(0, 0, rparams.width, rparams.height,
				0, 0, rparams.width, rparams.height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);

			// composite the resolved transparent layer over the opaque color buffer
			mFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
			mFramebuffer.drawBuffers(1, attachments); // color only
			glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		// wboit_composite outputs PREMULTIPLIED color (accum.rgb/accum.a already
		// carries alpha), so use ONE / ONE_MINUS_SRC_ALPHA for correct compositing.
		glBlendFunci(0, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			mWBOITCompositeProgram->use();
			mAccumResolveTex.bind(GL_TEXTURE0);
			mWBOITCompositeProgram->setInt("uAccum", 0);
			mRevealResolveTex.bind(GL_TEXTURE1);
			mWBOITCompositeProgram->setInt("uReveal", 1);
			mScreenQuad->draw();
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

		// Step 5.5: render "render-to-front" modules on top of everything.
		// These modules ignore depth written by other modules (rendered after
		// a depth-buffer clear), but still keep their own intra-module depth
		// relationships (depth test is enabled). Opaque and transparent
		// render-to-front modules are drawn in two separate passes.
		{
			// --- Opaque render-to-front modules (write depth) ---
			mFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
			mFramebuffer.drawBuffers(2, attachments); // color + index

			// Clear depth only (keep color/index) so render-to-front modules
			// are drawn on top of the existing scene.
			glClear(GL_DEPTH_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			glDepthMask(GL_TRUE);

			params.mode = GLRenderMode::COLOR;
			for (int i = 0; i < mRenderItems.size(); i++)
			{
				if (mRenderItems[i].node->isVisible()
					&& !mRenderItems[i].visualModule->isTransparent()
					&& mRenderItems[i].visualModule->varRenderToFront()->getValue())
				{
					params.index = i;
					float savedShadow = applyReceiveShadow(mRenderItems[i].visualModule);
					mRenderItems[i].visualModule->draw(params);
					params.light.mainLightShadow = savedShadow;
				}
			}

			// --- Transparent render-to-front modules (WBOIT, depth-tested
			//     against the opaque render-to-front depth written above) ---
			params.mode = GLRenderMode::TRANSPARENCY;

			mWBOITFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
			const unsigned int wboitBuffersRtf[] = { GL_COLOR_ATTACHMENT0, GL_NONE, GL_COLOR_ATTACHMENT2 };
			mWBOITFramebuffer.drawBuffers(3, wboitBuffersRtf);

			const float accumClearRtf[4] = { 0.f, 0.f, 0.f, 0.f };
			const float revealClearRtf[4] = { 1.f, 1.f, 1.f, 1.f };
			glClearBufferfv(GL_COLOR, 0, accumClearRtf);
			glClearBufferfv(GL_COLOR, 2, revealClearRtf);

			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);
			glDepthMask(GL_FALSE);

			glEnable(GL_BLEND);
			glBlendFunci(0, GL_ONE, GL_ONE);
			glBlendFunci(2, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);

			for (int i = 0; i < mRenderItems.size(); i++)
			{
				if (mRenderItems[i].node->isVisible()
					&& mRenderItems[i].visualModule->isTransparent()
					&& mRenderItems[i].visualModule->varRenderToFront()->getValue())
				{
					params.index = i;
					float savedShadow = applyReceiveShadow(mRenderItems[i].visualModule);
					mRenderItems[i].visualModule->draw(params);
					params.light.mainLightShadow = savedShadow;
				}
			}

			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);

			// resolve WBOIT G-buffers to single-sample targets
			mWBOITFramebuffer.bind(GL_READ_FRAMEBUFFER);
			mWBOITResolveFBO.bind(GL_DRAW_FRAMEBUFFER);
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glDrawBuffer(GL_COLOR_ATTACHMENT0);
			glBlitFramebuffer(0, 0, rparams.width, rparams.height,
				0, 0, rparams.width, rparams.height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
			glReadBuffer(GL_COLOR_ATTACHMENT2);
			glDrawBuffer(GL_COLOR_ATTACHMENT2);
			glBlitFramebuffer(0, 0, rparams.width, rparams.height,
				0, 0, rparams.width, rparams.height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);

			// composite the transparent render-to-front layer over the
			// current color buffer (which already contains the scene +
			// opaque render-to-front modules)
			mFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
			mFramebuffer.drawBuffers(1, attachments); // color only
			glDisable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunci(0, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
			mWBOITCompositeProgram->use();
			mAccumResolveTex.bind(GL_TEXTURE0);
			mWBOITCompositeProgram->setInt("uAccum", 0);
			mRevealResolveTex.bind(GL_TEXTURE1);
			mWBOITCompositeProgram->setInt("uReveal", 1);
			mScreenQuad->draw();
			glDisable(GL_BLEND);
			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
		}


		//// Step 6: draw to final framebuffer with fxaa filter
		{
			// restore previous framebuffer
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);

			if (bEnableFXAA)
			{
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glViewport(p.x, p.y, rparams.width, rparams.height);

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
					p.x, p.y, p.x + rparams.width, p.y + rparams.height,
					GL_COLOR_BUFFER_BIT, GL_LINEAR);
			}
		}

		glCheckError();
	}


	void GLRenderEngine::resizeFramebuffer(int w, int h, int samples)
	{
		// resize internal framebuffer
		mColorTex.resize(w, h, samples);
		mDepthTex.resize(w, h, samples);
		mIndexTex.resize(w, h, samples);
		mAccumTex.resize(w, h, samples);
		mRevealTex.resize(w, h, samples);

		mAccumResolveTex.resize(w, h);
		mRevealResolveTex.resize(w, h);

		mSelectIndexTex.resize(w, h);

		glCheckError();
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

		// save current framebuffer binding
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);

		// blit multisample framebuffer to regular framebuffer
		mFramebuffer.bind(GL_READ_FRAMEBUFFER);
		mSelectFramebuffer.bind(GL_DRAW_FRAMEBUFFER);
		glReadBuffer(GL_COLOR_ATTACHMENT1);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glBlitFramebuffer(x, y, x+w, y+h, x, y, x+w, y+h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

		// read pixels
		std::vector<glm::ivec4> indices(w * h);

		mSelectFramebuffer.bind(GL_READ_FRAMEBUFFER);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		//glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(x, y, w, h, GL_RGBA_INTEGER, GL_INT, indices.data());

		// restore current framebuffer binding
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		glCheckError();

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

	void GLRenderEngine::setMSAA(int samples)
	{
		// [0, 8]
		if (samples < 0) samples = 0;
		if (samples > 8) samples = 8;
		mMSAASamples = samples;
	}

	int GLRenderEngine::getMSAA() const
	{
		return mMSAASamples;
	}

	void GLRenderEngine::setFXAA(bool flag)
	{
		bEnableFXAA = flag;
	}

	int GLRenderEngine::getFXAA() const
	{
		return bEnableFXAA;
	}

}