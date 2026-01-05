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

#include "TriangleSet.h"

#include "Field/Color.h"

#include "Primitive/Primitive3D.h"

namespace dyno
{
	class Material : public Object
	{
	public:

		Material(){};
		~Material() override 
		{
			texColor.clear();
			texBump.clear();
			texORM.clear();
			texAlpha.clear();
			texEmissive.clear();
		};

		Color baseColor = Color::LightGray();
		float metallic = 0;
		float roughness = 0.5;
		float alpha = 1;
		float bumpScale = 1;
		float emissiveIntensity = 0;

		DArray2D<Vec4f> texColor;
		DArray2D<Vec4f> texBump;
		DArray2D<Vec4f> texORM;
		DArray2D<Vec4f> texAlpha;
		DArray2D<Vec4f> texEmissive;
	};


	class Shape : public Object
	{
	public:
		Shape() {};
		Shape(const Shape& other) 
		{
			this->vertexIndex.assign(other.vertexIndex);
			this->normalIndex.assign(other.normalIndex);
			this->texCoordIndex.assign(other.texCoordIndex);
			this->boundingBox = other.boundingBox;
			this->boundingTransform = other.boundingTransform;
			this->material = other.material;
		}
		~Shape() override { clear(); };
		void clear() 
		{
			vertexIndex.clear();
			normalIndex.clear();
			texCoordIndex.clear();
			material = nullptr;
		}
		DArray<TopologyModule::Triangle> vertexIndex;
		DArray<TopologyModule::Triangle> normalIndex;
		DArray<TopologyModule::Triangle> texCoordIndex;

		TAlignedBox3D<Real> boundingBox;
		Transform3f boundingTransform;

		std::shared_ptr<Material> material = nullptr;

	};


	class Geometry : public Object
	{
	public:
		Geometry() {};
		~Geometry() { clear(); };

		void clear() 
		{
			mVertices.clear();
			mNormals.clear();
			mTexCoords.clear();
			mShapeIds.clear();
		}

		void assign(std::shared_ptr<Geometry> dataPtr)
		{
			if (dataPtr) 
			{
				mVertices.assign(dataPtr->vertices());
				mNormals.assign(dataPtr->normals());
				mTexCoords.assign(dataPtr->texCoords());
				mShapeIds.assign(dataPtr->shapeIds());
			}
		}

		DArray<Vec3f>& vertices() { return mVertices; }
		DArray<Vec3f>& normals() { return mNormals; }
		DArray<Vec2f>& texCoords() { return mTexCoords; }
		DArray<uint>& shapeIds() { return mShapeIds; }

	private:
		DArray<Vec3f> mVertices;
		DArray<Vec3f> mNormals;
		DArray<Vec2f> mTexCoords;
		DArray<uint> mShapeIds;
	};

	class TextureMesh : public TopologyModule
	{
	public:
		TextureMesh();
		~TextureMesh() override;

		std::shared_ptr<Geometry> geometry();

		std::vector<std::shared_ptr<Shape>>& shapes() { return mShapes; }

		void merge(const std::shared_ptr<TextureMesh> texMesh01, const std::shared_ptr<TextureMesh> texMesh02);

		void clear();

		void safeConvert2TriangleSet(TriangleSet<DataType3f>& triangleSet);

		void convert2TriangleSet(TriangleSet<DataType3f>& triangleSet);

		std::vector<Vec3f> updateTexMeshBoundingBox();

		template<typename Vec3f>
		void transPoint2Vertices(
			DArray<Vec3f>& pAttribute,
			DArray<Vec3f>& vAttribute,
			DArrayList<int>& contactList
		);

	private:
		std::shared_ptr<Geometry> mMeshData = NULL;
		std::vector<std::shared_ptr<Shape>> mShapes;
	};

	
};