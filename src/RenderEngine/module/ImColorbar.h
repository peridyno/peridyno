/**
 * Copyright 2017-2021 Xukun LUO
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

#include "GLVisualModule.h"
#include <imgui.h>
#include "ui/imgui_extend.h"

// #include "GLCudaBuffer.h"
// #include "gl/VertexArray.h"
// #include "gl/Program.h"

namespace dyno
{
	class ImColorbar : public GLVisualModule
	{
		DECLARE_CLASS(ImColorbar)
	public:

		ImColorbar();
		~ImColorbar() override;

		void setCoord(ImVec2 coord);

		ImVec2 getCoord() const;

	public:
		DEF_ARRAY_IN(Vec3f, Color, DeviceType::GPU, "");
		// DEF_ARRAY_IN(int, Value, DeviceType::GPU, "");

	protected:
		virtual void paintGL(RenderMode mode) override;
		virtual void updateGL() override;
		virtual bool initializeGL() override;

	private:

		const int* 				mValue;
		const Vec3f*  			mColor;
		int						mNum = 0;
        ImVec2              	mCoord = ImVec2(0,0);
	};
};
