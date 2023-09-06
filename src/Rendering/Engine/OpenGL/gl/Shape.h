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

#include "Object.h"
#include "Platform.h"

#include "GPUBuffer.h"

#include "Material.h"

#include "Topology/TriangleSet.h"

namespace gl
{
	class Shape : public Object
	{
	public:
		Shape();
		~Shape() override;

		virtual void create();
		virtual void release();

		void update();

		void updateGL();

		dyno::DArray<dyno::TopologyModule::Triangle> vertexIndex;
		dyno::DArray<dyno::TopologyModule::Triangle> normalIndex;
		dyno::DArray<dyno::TopologyModule::Triangle> texCoordIndex;

		gl::XBuffer<dyno::TopologyModule::Triangle>		glVertexIndex;
		gl::XBuffer<dyno::TopologyModule::Triangle>		glNormalIndex;
		gl::XBuffer<dyno::TopologyModule::Triangle>		glTexCoordIndex;

		std::shared_ptr<Material> material = nullptr;

	private:
		bool mInitialized = false;
	};
};