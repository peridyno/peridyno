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

#include "GLVisualModule.h"

#include "GraphicsObject/VertexArray.h"
#include "GraphicsObject/Shader.h"
#include "GraphicsObject/GLTextureMesh.h"

#include "Topology/TextureMesh.h"
#include "GLPhotorealisticRender.h"

#ifdef CUDA_BACKEND
#include "ConstructTangentSpace.h"
#endif

namespace dyno
{
	class GLShapeRender : public GLPhotorealisticRender
	{
		DECLARE_CLASS(GLShapeRender)
	public:
		GLShapeRender();

	public:
		virtual std::string caption() override { return "GLShapeRender"; };

	public:
		DEF_VAR(std::vector<uint>, Shapes, std::vector<uint>{0}, "");

	protected:

		virtual void paintGL(const RenderParams& rparams) override;

	protected:
		
	};
};