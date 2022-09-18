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

#include "gl/Shader.h"
#include "gl/Mesh.h"

namespace dyno
{
	class FXAA
	{
	public:
		FXAA();
		~FXAA();

		void initialize();
		void apply(int width, int height);

	public:
		float RelativeContrastThreshold;
		float HardContrastThreshold;
		float SubpixelBlendLimit;
		float SubpixelContrastThreshold;
		int EndpointSearchIterations;
		bool UseHighQualityEndpoints;

	private:

		gl::Program		mShaderProgram;
		gl::Mesh		mScreenQuad;
	};
}