/**
 * Copyright 2024 Xiaowei He
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

#include "GPUBuffer.h"
#include "GraphicsObject.h"

#include "GraphicsObject/GPUTexture.h"

#include "Topology/TextureMesh.h"

#include <glm/glm.hpp>

namespace dyno
{
	// simple PBR material model
	class GLMaterial : public GraphicsObject
	{
	public:

		GLMaterial();
		~GLMaterial() override;
		void create() override;
		void release() override;

		void updateGL();

	public:
		Vec3f baseColor = { 0.8f, 0.8f, 0.8f };
		
		float metallic = 0.0f;
		float roughness = 1.0f;
		float alpha = 1.0f;

		float bumpScale = 1.f;

		// color texture
		XTexture2D<dyno::Vec4f> texColor;
		XTexture2D<dyno::Vec4f> texBump;
		XTexture2D<dyno::Vec4f> texORM;
		XTexture2D<dyno::Vec4f> texAlpha;

		bool mInitialized = false;
	};

	class GLShape : public GraphicsObject
	{
	public:
		GLShape();
		~GLShape() override;

		void create() override;
		void release() override;

		void updateGL();

		XBuffer<dyno::TopologyModule::Triangle>		glVertexIndex;
		XBuffer<dyno::TopologyModule::Triangle>		glNormalIndex;
		XBuffer<dyno::TopologyModule::Triangle>		glTexCoordIndex;

		glm::mat4 transform;

		std::shared_ptr<GLMaterial> material = nullptr;

	private:
		bool mInitialized = false;
	};

	class GLTextureMesh : public GraphicsObject
	{
	public:
		GLTextureMesh();
		~GLTextureMesh() override;

		void create() final;
		void release() final;

		void load(const std::shared_ptr<TextureMesh> mesh);

		void updateGL();

		inline XBuffer<Vec3f>& vertices() { return mVertices; }
		inline XBuffer<Vec3f>& normals() { return mNormal; }
		inline XBuffer<Vec2f>& texCoords() { return mTexCoord; }

		inline std::vector<std::shared_ptr<GLShape>>& shapes() { return mShapes; }

	private:
		XBuffer<Vec3f> mVertices;
		XBuffer<Vec3f> mNormal;
		XBuffer<Vec2f> mTexCoord;

		std::vector<std::shared_ptr<GLShape>> mShapes;

		bool mInitialized = false;
	};
};