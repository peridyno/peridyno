/**
 * Copyright 2023 Xiaowei He
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

#include "Module/ComputeModule.h"

#include "Topology/TriangleSet.h"
#include "gl/Shape.h"

namespace dyno
{
	class ConstructTangentSpace : public ComputeModule
	{
		DECLARE_CLASS(GLSurfaceVisualModule)
	public:
		ConstructTangentSpace();
		~ConstructTangentSpace() override;

	public:
		DEF_ARRAY_IN(Vec3f, Vertex, DeviceType::GPU, "");

		DEF_ARRAY_IN(Vec3f, Normal, DeviceType::GPU, "");

		DEF_ARRAY_IN(Vec2f, TexCoord, DeviceType::GPU, "");

		DEF_INSTANCES_IN(gl::Shape, Shape, "");

		DEF_ARRAY_OUT(Vec3f, Normal, DeviceType::GPU, "");

		DEF_ARRAY_OUT(Vec3f, Tangent, DeviceType::GPU, "");

		DEF_ARRAY_OUT(Vec3f, Bitangent, DeviceType::GPU, "");

	protected:
		void compute() override;

	private:
	};
};