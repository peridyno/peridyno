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

#include <Vector.h>
#include <RenderParams.h>

namespace dyno
{
	class BBoxRenderer;
	class GroundRenderer;
	class BackgroundRenderer;
	class GLRenderHelper
	{
	public:
		GLRenderHelper();
		~GLRenderHelper();

		void drawGround(const RenderParams& rparams, 
			float planeScale, float rulerScale = 1.f,
			dyno::Vec4f planeColor = { 0.3, 0.3, 0.3, 0.5 },
			dyno::Vec4f rulerColor = { 0.1, 0.1, 0.1, 0.5 });
		void drawBBox(const RenderParams& rparams, Vec3f pmin, Vec3f pmax, int type = 0);
		void drawBackground(Vec3f color0, Vec3f color1);

	private:
		BBoxRenderer*			mBBoxRenderer = NULL;
		GroundRenderer*			mGroundRenderer = NULL;
		BackgroundRenderer*		mBackgroundRenderer = NULL;
	};
}

