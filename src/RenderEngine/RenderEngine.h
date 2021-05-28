#pragma once

#include "GLVertexArray.h"
#include "GLShader.h"
#include "GLTexture.h"
#include "RenderParams.h"

#include <vector>

class ShadowMap;
class RenderHelper;
namespace dyno
{
	class SceneGraph;
	class GLVisualModule;
	class RenderTarget;
	class RenderEngine
	{
	public:
		RenderEngine();
		~RenderEngine();

		void initialize();
		void draw(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams);

	private:
		void initUniformBuffers();

		void renderSetup(dyno::SceneGraph* scene, RenderTarget* target, const RenderParams& rparams);
		void updateShadowMap(const RenderParams&);

		// render pass
		void renderBackground(RenderTarget* target, const RenderParams& rparams);
		void renderOpacity(RenderTarget* target, const RenderParams& rparams);
		void renderTransparency(RenderTarget* target, const RenderParams& rparams);
		void renderFluid(RenderTarget* target, const RenderParams& rparams);
		void renderPostprocess(RenderTarget* target, const RenderParams& rparams);

		// surface material
		void setMaterial(dyno::GLVisualModule* m);
		void renderModule(dyno::GLVisualModule* m, unsigned int subroutine);

	private:
		// uniform buffer for matrices
		GLBuffer mTransformUBO;
		GLBuffer mShadowMapUBO;
		GLBuffer mLightUBO;
		GLBuffer mMaterialUBO;

		GLBuffer mSSAOKernelUBO;
		GLTexture2D mSSAONoiseTex;

		GLShaderProgram mSurfaceProgram;
		GLShaderProgram mPointProgram;

		GLShaderProgram mPBRShadingProgram;
		GLShaderProgram mSSAOProgram;
		GLShaderProgram mFXAAProgram;

		// transparency blend
		GLShaderProgram mBlendProgram;

		// fluid
		GLShaderProgram mFluidProgram;
		GLShaderProgram mFluidFilterProgram;
		GLShaderProgram mFluidBlendProgram;

		GLMesh			mScreenQuad;

		ShadowMap* mShadowMap;
		RenderHelper* mRenderHelper;

	private:
		std::vector<dyno::GLVisualModule*> mRenderQueue;

		void enqueue(dyno::GLVisualModule* m) {
			mRenderQueue.push_back(m);
		}
		friend class DrawAct2;
	};
};
