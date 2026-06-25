/**
 * Copyright 2017-2021 Jian SHI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <memory>
#include <vector>

#include <RenderEngine.h>

#include "GraphicsObject/Buffer.h"
#include "GraphicsObject/Texture.h"
#include "GraphicsObject/Framebuffer.h"
#include "GraphicsObject/Shader.h"
#include "GraphicsObject/Mesh.h"


namespace dyno
{
	class SSAO;
	class FXAA;
	class Envmap;
	class ShadowMap;
	class GLRenderHelper;
	class GLVisualModule;
	class SceneGraph;

	class GLRenderEngine : public RenderEngine
	{
	public:
		GLRenderEngine();
		~GLRenderEngine();
			   
		virtual void initialize() override;
		virtual void terminate() override;

		virtual void draw(dyno::SceneGraph* scene, const RenderParams& rparams) override;

		virtual std::string name() const override;

		// get the selected nodes on given rect area
		Selection select(int x, int y, int w, int h) override;

		// use MSAA samples
		void setMSAA(int samples);
		int  getMSAA() const;

		void setFXAA(bool flag);
		int getFXAA() const;

		void setShadowMapSize(int size);
		int  getShadowMapSize() const;

		void setShadowBlurIters(int iters);
		int  getShadowBlurIters() const;

		void setDefaultEnvmap() override;
		void setEnvmap(const std::string& path);

		void setEnvStyle(EEnvStyle style) override;

		inline std::string getEnvmapFilePath() { return mEnvmapFilePath; }

		int  getShadowMapSize();
		void updateShadowMapAttribute()override;
	protected:
		void createFramebuffer();
		void resizeFramebuffer(int w, int h, int samples);
		void setupTransparencyPass();
		void updateRenderItems(dyno::SceneGraph* scene);

	private:

		// objects to render
		struct RenderItem {
			std::shared_ptr<Node>			node;
			std::shared_ptr<GLVisualModule> visualModule;

			bool operator==(const RenderItem& item) {
				return node == item.node && visualModule == item.visualModule;
			}
		};

		std::vector<RenderItem> mRenderItems;

	protected:

		//Texture2DMultiSample	mColorCorrectTex;
		Program* mPostProcessProgram;

		// internal framebuffer
		Framebuffer				mFramebuffer;
		Texture2DMultiSample	mColorTex;
		Texture2DMultiSample	mDepthTex;
		Texture2DMultiSample	mIndexTex;			// indices for object/mesh/primitive etc.

		// non-multisample framebuffer for select
		Framebuffer				mSelectFramebuffer;
		Texture2D				mSelectIndexTex;

		// for linked-list OIT
		// Per-pixel fragment-node budget for Order-Independent Transparency.
		// Memory = 32 bytes/node, so 32M nodes = 1 GB. IMPORTANT: this MUST stay in
		// sync with `uMaxNodes` in shader/transparency.glsl (the shader's write guard);
		// after changing that .glsl the engine must be rebuilt so the embedded SPIR-V
		// is regenerated. See setupTransparencyPass()/Step 4 overflow check.
		const int				MAX_OIT_NODES = 1024 * 1024 * 32;
		Buffer					mFreeNodeIdx;
		Buffer					mLinkedListBuffer;
		Texture2DMultiSample	mHeadIndexTex;
		Program*				mBlendProgram;
		// OIT overflow telemetry: when set, Step 4 reads back the free-node counter
		// and logs once on overflow (a fragment-node-pool overflow silently drops
		// transparent fragments -> view-dependent ordering artifacts). The readback
		// forces a GPU sync, so this can be disabled for max throughput.
		// The readback forces a GPU sync (stall ~= the OIT build-pass GPU time), so it is
		// sampled only every mOITCheckInterval frames rather than every frame. Overflow is
		// a slow-changing condition, so a few-frame detection latency is fine and keeps the
		// stall cost negligible. Set mReportOITOverflow=false to disable entirely.
		bool					mReportOITOverflow = true;
		int						mOITCheckInterval = 30;		// frames between counter readbacks
		int						mOITFrameCounter = 0;
		bool					mOITOverflowState = false;	// last-sample overflow state (edge-triggered logging)

		GLRenderHelper*			mRenderHelper;
		ShadowMap*				mShadowMap = NULL;

		// anti-aliasing
		
		// MSAA samples
		int						mMSAASamples = 4;

		// FXAA
		bool					bEnableFXAA = false;
		FXAA*					mFXAAFilter;

		//ShadowType
		int						mShadowType = 2;

		// Envmap
		std::string				mEnvmapFilePath = getAssetPath() + "textures/hdr/venice_dawn_1_4k.hdr";
		Envmap*					mEnvmap = NULL;
		
		Mesh* mScreenQuad = 0;
	};
};
