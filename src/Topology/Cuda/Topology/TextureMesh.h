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

#include "Module/TopologyModule.h"
#include "Primitive/Primitive3D.h"


namespace dyno
{
	// simple PBR material model
	class Material : public Object
	{
	public:
		Material() {};
		~Material() override {
			texColor.clear();
			texBump.clear();
		};

	public:

		Vec3f baseColor = { 0.8f, 0.8f, 0.8f };
		
		float metallic  = 0.0f;
		float roughness = 1.0f;
		float alpha = 1.0f;

		DArray2D<Vec4f> texColor;

		DArray2D<Vec4f> texBump;
		float bumpScale = 1.f;
	};

	class Shape : public Object
	{
	public:
		Shape() {};
		~Shape() override {
			vertexIndex.clear();
			normalIndex.clear();
			texCoordIndex.clear();
		};

		DArray<TopologyModule::Triangle> vertexIndex;
		DArray<TopologyModule::Triangle> normalIndex;
		DArray<TopologyModule::Triangle> texCoordIndex;

		TAlignedBox3D<Real> boundingBox;
		Transform3f boundingTransform;

		std::shared_ptr<Material> material = nullptr;

	};

	class TextureMesh : public TopologyModule
	{
	public:
		TextureMesh();
		~TextureMesh() override;

		DArray<Vec3f>& vertices() { return mVertices; }
		DArray<Vec3f>& normals() { return mNormals; }
		DArray<Vec2f>& texCoords() { return mTexCoords; }
		DArray<uint>& shapeIds() { return mShapeIds; }

		std::vector<std::shared_ptr<Shape>>& shapes() { return mShapes; }
		std::vector<std::shared_ptr<Material>>& materials() { return mMaterials; }


		virtual void clear() 
		{
			mVertices.clear(); 
			mNormals.clear();
			mTexCoords.clear();
			mMaterials.clear();
			mMaterials.clear();
			mShapes.clear();
		}

	private:
		DArray<Vec3f> mVertices;
		DArray<Vec3f> mNormals;
		DArray<Vec2f> mTexCoords;
		DArray<uint> mShapeIds;

		std::vector<std::shared_ptr<Material>> mMaterials;
		std::vector<std::shared_ptr<Shape>> mShapes;
	};
};