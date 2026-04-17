#include "GLMeshRenderEngine.h"

// Visual modules for rendering
#include "Backend/Cuda/Module/GLSurfaceVisualModule.h"

//
#include "GLRenderHelper.h"
#include "GLVisualModule.h"

#include "ShadowMap.h"
#include "SSAO.h"
#include "FXAA.h"
#include "Envmap.h"

// dyno
#include "SceneGraph.h"
#include "Action.h"

// GLM

#include <OrbitCamera.h>
#include <TrackballCamera.h>
#include <unordered_set>
#include <memory>

#include "screen.vert.h"
#include "blend.frag.h"
#include "postprocess.frag.h"
#include "surface.frag.h"
#include "cuda/Module/GLPhotorealisticRender.h"
#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

namespace dyno
{
	GLMeshRenderEngine::GLMeshRenderEngine()
	{
		surfaceRenderModule = std::make_shared<GLSurfaceVisualModule>();
		realisticRenderModule = std::make_shared<GLPhotorealisticRender>();
		transparencyRealisticModule = std::make_shared<GLPhotorealisticRender>();
	}

	GLMeshRenderEngine::~GLMeshRenderEngine()
	{
		this->terminate();
		if (surfaceRenderModule)
			surfaceRenderModule->release();
		if (realisticRenderModule)
			realisticRenderModule->release();
		if (transparencyRealisticModule)
			transparencyRealisticModule->release();
	}

	void GLMeshRenderEngine::renderMesh(const std::vector<FInstance<TextureMesh>*>& texmeshs, const std::vector<FInstance<TriangleSet<DataType3f>>*>& triangles, const RenderParams& rparams, bool renderTransparency)
	{
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

		//// update shadow map
		//mShadowMap->update(nullptr, rparams);

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
		if(false)
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

		//mShadowMap->bind();

		mEnvmap->setScale(enmapScale);
		mEnvmap->bindIBL();

		// Step 2: render opacity objects
		{
			params.mode = GLRenderMode::COLOR;

			// Render TextureMesh instances
			for (size_t i = 0; i < texmeshs.size(); i++)
			{
				if (texmeshs[i]->isEmpty())
					continue;

				auto textureMesh = texmeshs[i]->constDataPtr();
				if (!textureMesh) continue;

				// Create a surface visual module for rendering

				texmeshs[i]->connect(realisticRenderModule->inTextureMesh());
				realisticRenderModule->update();

				// Render the mesh
				if (realisticRenderModule->isVisible())
				{
					params.index = static_cast<int>(i);
					realisticRenderModule->draw(params);
				}
			}

			if (!renderTransparency) 
			{
				// Render TriangleSet instances
				for (size_t i = 0; i < triangles.size(); i++)
				{
					if (triangles[i]->isEmpty())
						continue;

					auto triangleSet = triangles[i]->constDataPtr();
					if (!triangleSet) continue;

					triangles[i]->connect(surfaceRenderModule->inTriangleSet());
					// Create a surface visual module for rendering
					surfaceRenderModule->update();
					surfaceRenderModule->varAlpha()->setValue(1);

					// Render the mesh
					if (surfaceRenderModule->isVisible() && !surfaceRenderModule->isTransparent())
					{
						params.index = static_cast<int>(texmeshs.size() + i);
						surfaceRenderModule->draw(params);
					}
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

			// Render TextureMesh instances
			if (renderTransparency) 
			{
				for (size_t i = 0; i < texmeshs.size(); i++)
				{
					if (texmeshs[i]->isEmpty())
						continue;

					auto textureMesh = texmeshs[i]->constDataPtr();

					// Create a surface visual module for rendering

					texmeshs[i]->connect(transparencyRealisticModule->inTextureMesh());
					transparencyRealisticModule->varAlpha()->setValue(0.15);
					transparencyRealisticModule->update();

					// Render the mesh
					if (transparencyRealisticModule->isVisible())
					{
						params.index = static_cast<int>(texmeshs.size() + triangles.size() + i);
						transparencyRealisticModule->draw(params);
					}
				}

				// Render TriangleSet instances
				for (size_t i = 0; i < triangles.size(); i++)
				{
					if (triangles[i]->isEmpty())
						continue;

					auto triangleSet = triangles[i]->constDataPtr();
					if (!triangleSet) continue;

					triangles[i]->connect(surfaceRenderModule->inTriangleSet());

					surfaceRenderModule->update();
					surfaceRenderModule->varAlpha()->setValue(0.2);
					// Render the mesh
					if (surfaceRenderModule->isVisible() && surfaceRenderModule->isTransparent())
					{
						params.index = static_cast<int>(2 * texmeshs.size() + triangles.size() + i);
						surfaceRenderModule->draw(params);
					}
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


		//// Step 6: draw to final framebuffer with fxaa filter
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
					GL_COLOR_BUFFER_BIT, GL_LINEAR);
			}
		}

		glCheckError();
	}

	std::string GLMeshRenderEngine::name() const
	{
		return std::string("GL Mesh Render Engine");
	}

}
