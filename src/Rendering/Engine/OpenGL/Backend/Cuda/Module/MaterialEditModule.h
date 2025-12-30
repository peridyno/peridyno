/**
 * Copyright 2025 Yuzhong Guo
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
#include "Node.h"
#include "Module/TopologyMapping.h"
#include "Module/KeyboardInputModule.h"
#include "Topology/TextureMesh.h"
#include "Topology/TriangleSet.h"
#include "Module/ComputeModule.h"
#include <Field/Color.h>
#include "Field/FilePath.h"
#include "MaterialManager.h"

namespace dyno
{

	class ColorCorrect : public MaterialManagedModule
	{
		DECLARE_CLASS(ColorCorrect)
	public:

		ColorCorrect();
		ColorCorrect(std::shared_ptr<ColorCorrect> other);
		~ColorCorrect() {};

		void onFieldChanged();
		void updateImpl() override { onFieldChanged(); };
		std::string caption() override { return "ColorCorrect"; }

		void initial();

		DEF_ARRAY2D_IN(Vec4f, Texture, DeviceType::GPU, "ColorTexture");

		//Color Correct
		DEF_VAR(float, Saturation, 1, "");
		DEF_VAR(float, HUEOffset, 0, "");
		DEF_VAR(float, Contrast, 1, "");
		DEF_VAR(float, Brightness, 1, "");
		DEF_VAR(float, Gamma, 1, "");
		DEF_VAR(Color, TintColor, Color(0.5), "");
		DEF_VAR(float, TintIntensity, 0, "");
		DEF_ARRAY2D_OUT(Vec4f, Texture, DeviceType::GPU, "");

		//
		DEF_VAR_IN(float, Saturation, "");
		DEF_VAR_IN(float, HUEOffset,"");
		DEF_VAR_IN(float, Contrast, "");
		DEF_VAR_IN(float, Brightness, "");
		DEF_VAR_IN(float, Gamma, "");
		DEF_VAR_IN(float, TintIntensity, "");
		DEF_VAR_IN(Color, TintColor, "");

	protected:
		virtual std::shared_ptr<MaterialManagedModule> clone() const override;
		
	};

	class GrayscaleCorrect : public MaterialManagedModule
	{
		DECLARE_CLASS(GrayscaleCorrect)
	public:

		GrayscaleCorrect();
		GrayscaleCorrect(std::shared_ptr<GrayscaleCorrect> other);
		~GrayscaleCorrect() {};

		void onFieldChanged();
		void updateImpl() override { onFieldChanged(); };
		std::string caption() override { return "GrayscaleCorrect"; }
		void initial();
		DEF_ARRAY2D_IN(float, GrayscaleTexture, DeviceType::GPU, "GrayscaleTexture");

		DEF_VAR(float, Contrast, 1, "");
		DEF_VAR(float, Brightness, 1, "");
		DEF_VAR(float, Gamma, 1, "");

		DEF_ARRAY2D_OUT(float, GrayscaleTexture, DeviceType::GPU, "");

	protected:
		virtual std::shared_ptr<MaterialManagedModule> clone() const override;

	};

	template<typename TDataType>
	class AssignTextureMeshMaterial : public ComputeModule
	{
		DECLARE_TCLASS(AssignTextureMeshMaterial, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::Transform<Real, 3> Transform;

		AssignTextureMeshMaterial();
		~AssignTextureMeshMaterial() { removeMaterialReference(); }

		void updateAssign();
		void compute() override;

		std::weak_ptr<Module> getSelfPtr();

		std::string caption() override { return "AssignTextureMeshMaterial"; }

		//Texture
		DEF_VAR(uint, ShapeIndex, 0, "");
		DEF_VAR(std::string, MaterialName, "", "");
		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
		DEF_INSTANCE_OUT(TextureMesh, TextureMesh, "");

	public:
		std::weak_ptr<CustomMaterial> varMat;

	protected:
		void removeMaterialReference();

	};

	class BreakTexture : public MaterialManagedModule
	{
		DECLARE_CLASS(BreakTexture)

	public:

		BreakTexture();
		BreakTexture(std::shared_ptr<BreakTexture> other);
		~BreakTexture() {}

		void onFieldChanged();
		void updateImpl() override { onFieldChanged(); };
		std::string caption() override { return "BreakTexture"; }
		void initial();
		//Texture
		DEF_ARRAY2D_IN(Vec4f, Texture, DeviceType::GPU, "ColorTextureOverride");

		DEF_ARRAY2D_OUT(float, R, DeviceType::GPU, "R");
		DEF_ARRAY2D_OUT(float, G, DeviceType::GPU, "G");
		DEF_ARRAY2D_OUT(float, B, DeviceType::GPU, "B");
		DEF_ARRAY2D_OUT(float, A, DeviceType::GPU, "A");

	protected:
		virtual std::shared_ptr<MaterialManagedModule> clone() const override;

	};

	class MakeTexture : public MaterialManagedModule
	{
		DECLARE_CLASS(MakeTexture)

	public:

		MakeTexture();
		MakeTexture(std::shared_ptr<MakeTexture> other);
		~MakeTexture() {}

		void onFieldChanged();
		void updateImpl() override { onFieldChanged(); };
		std::string caption() override { return "MakeTexture"; }
		void initial();
		//Texture
		DEF_VAR(float, R, 0, "");
		DEF_VAR(float, G, 0, "");
		DEF_VAR(float, B, 0, "");
		DEF_VAR(float, A, 1, "");

		DEF_ARRAY2D_IN(float, R, DeviceType::GPU, "R");
		DEF_ARRAY2D_IN(float, G, DeviceType::GPU, "G");
		DEF_ARRAY2D_IN(float, B, DeviceType::GPU, "B");
		DEF_ARRAY2D_IN(float, A, DeviceType::GPU, "A");

		DEF_ARRAY2D_OUT(Vec4f, Texture, DeviceType::GPU, "ColorTextureOverride");

	protected:
		virtual std::shared_ptr<MaterialManagedModule> clone() const override;
	};

	class MixTexture : public MaterialManagedModule
	{
		DECLARE_CLASS(MixTexture)

	public:

		MixTexture();
		~MixTexture() {}

		void onFieldChanged();
		void updateImpl() override { onFieldChanged(); };
		std::string caption() override { return "MixTexture"; }

		//Texture
		DEF_VAR(float, Weight, 0, "");

		DEF_ARRAY2D_IN(Vec4f, TextureA, DeviceType::GPU, "TextureA");
		DEF_ARRAY2D_IN(Vec4f, TextureB, DeviceType::GPU, "TextureB");
		DEF_ARRAY2D_IN(float, FloatA, DeviceType::GPU, "FloatA");
		DEF_ARRAY2D_IN(float, FloatB, DeviceType::GPU, "FloatB");
		DEF_ARRAY2D_IN(float, Mask, DeviceType::GPU, "TextureC");

		DEF_ARRAY2D_OUT(Vec4f, Texture, DeviceType::GPU, "ColorTextureOverride");
		DEF_ARRAY2D_OUT(float, Float, DeviceType::GPU, "ColorTextureOverride");

	};

	class MatInput : public KeyboardInputModule
	{
		DECLARE_CLASS(MatInput)

	public:

		MatInput() { this->outValue()->setValue(0); };
		~MatInput() {}

		DEF_VAR_OUT(float,Value,"KeyValue");
	protected:
		virtual void onEvent(PKeyboardEvent event)override;

	};

//	class MaterialGroup : public TopologyModule
//	{
//	public:
//		MaterialGroup() {};
//		~MaterialGroup() = default;
//		std::shared_ptr<std::vector<std::shared_ptr<Material>>> Materials;
//
//		FInstance<TextureMesh>* TexMeshSource = NULL;
//	};
//
//	class ShapeGroup : public TopologyModule
//	{
//	public:
//		ShapeGroup() {};
//		~ShapeGroup() = default;
//
//		std::shared_ptr<std::vector<std::shared_ptr<Shape>>> Shapes;
//		FInstance<TextureMesh>* TexMeshSource = NULL;
//	};
//
//
//	template<typename TDataType>
//	class BreakTextureMesh : public ComputeModule
//	{
//		DECLARE_TCLASS(BreakTextureMesh, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		BreakTextureMesh();
//		~BreakTextureMesh();
//		void compute() override;
//		std::string caption() override { return "BreakTextureMeshModule"; }
//		void breakTexMesh();
//
//		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
//
//		DEF_ARRAY_OUT(Vec3f, Vertices, DeviceType::GPU, "");
//		DEF_ARRAY_OUT(Vec3f, Normals, DeviceType::GPU, "");
//		DEF_ARRAY_OUT(Vec2f, TexCoords, DeviceType::GPU, "");
//		DEF_ARRAY_OUT(uint, ShapeIds, DeviceType::GPU, "");
//
//		DEF_INSTANCE_OUT(MaterialGroup, MaterialGroup, "");
//		DEF_INSTANCE_OUT(ShapeGroup, ShapeGroup, "");
//
//	protected:
//
//	};
//
//	template<typename TDataType>
//	class MakeTextureMesh : public ComputeModule
//	{
//		DECLARE_TCLASS(MakeTextureMesh, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		MakeTextureMesh();
//		~MakeTextureMesh();
//		void compute() override;
//
//		void onFieldChanged();
//		std::string caption() override { return "MakeTextureMesh"; }
//
//		DEF_INSTANCE_OUT(TextureMesh, TextureMesh, "");
//
//		DEF_ARRAY_IN(Vec3f, Vertices, DeviceType::GPU, "");
//		DEF_ARRAY_IN(Vec3f, Normals, DeviceType::GPU, "");
//		DEF_ARRAY_IN(Vec2f, TexCoords, DeviceType::GPU, "");
//		DEF_ARRAY_IN(uint, ShapeIds, DeviceType::GPU, "");
//
//		DEF_INSTANCE_IN(MaterialGroup, MaterialGroup, "");
//		DEF_INSTANCE_IN(ShapeGroup, ShapeGroup, "");
//
//	protected:
//		
//	};
//
//	template<typename TDataType>
//	class GetMaterialFromGroup : public ComputeModule
//	{
//		DECLARE_TCLASS(GetMaterialFromGroup, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		GetMaterialFromGroup() ;
//		~GetMaterialFromGroup() {};
//		void compute() override;
//		void onIndexChanged();
//		void updateMaterial();
//		std::string caption() override { return "GetMaterialFromGroup"; }
//
//		DEF_VAR(uint, Index, 0, "");
//
//		DEF_VAR(Color, Color, Color(1),"");
//		DEF_VAR(float, Metallic, 0, "");
//		DEF_VAR(float, Roughness, 0, "");
//		DEF_VAR(float, Alpha, 0, "");
//		DEF_VAR(float, BumpScale, 0, "");
//
//		DEF_INSTANCE_IN(MaterialGroup, MaterialGroup, "");
//		DEF_INSTANCE_OUT(Material, Material, "");
//
//		DEF_ARRAY2D_OUT(Vec4f, TexColor, DeviceType::GPU, "TexColor");
//		DEF_ARRAY2D_OUT(Vec4f, TexBump, DeviceType::GPU, "texBump");
//		DEF_ARRAY2D_OUT(Vec4f, TexORM, DeviceType::GPU, "texORM");
//		DEF_ARRAY2D_OUT(Vec4f, TexAlpha, DeviceType::GPU, "texAlpha");
//
//	protected:
//
//	private:
//		std::shared_ptr<Material> mMaterial = NULL;
//	};
//
//	template<typename TDataType>
//	class GetShapeFromGroup : public ComputeModule
//	{
//		DECLARE_TCLASS(GetShapeFromGroup, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		GetShapeFromGroup();
//		~GetShapeFromGroup() {};
//
//		void onFieldChanged();
//		void compute() override { onFieldChanged(); };
//		std::string caption() override { return "GetShapeFromGroup"; }
//
//		DEF_VAR(uint, Index, 0, "");
//
//		DEF_INSTANCE_IN(ShapeGroup, ShapeGroup, "");
//		DEF_INSTANCE_IN(Material, MaterialOverride, "");
//		DEF_INSTANCE_OUT(Shape, Shape, "");
//
//	protected:
//
//	private:
//		std::shared_ptr<Material> mMaterial = NULL;
//	};
//
//
//	template<typename TDataType>
//	class MaterialBinder : public ComputeModule
//	{
//		DECLARE_TCLASS(MaterialBinder, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		MaterialBinder();
//		~MaterialBinder() {};
//
//		void onFieldChanged();
//		void compute() override { onFieldChanged(); };
//		std::string caption() override { return "MaterialBinder"; }
//
//		DEF_INSTANCE_IN(Shape, Shape, "");
//		DEF_INSTANCE_IN(Material, Material, "");
//		DEF_INSTANCE_OUT(Shape, Shape, "");
//
//	protected:
//
//	};
//
//	template<typename TDataType>
//	class MakeShapeGroup : public ComputeModule
//	{
//		DECLARE_TCLASS(MakeShapeGroup, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		MakeShapeGroup();
//		~MakeShapeGroup() {};
//
//		void onFieldChanged();
//		void compute() override { onFieldChanged(); };
//		std::string caption() override { return "MakeShapeGroup"; }
//
//		DEF_INSTANCE_IN(Shape, Shape0, "");
//		DEF_INSTANCE_IN(Shape, Shape1, "");
//		DEF_INSTANCE_IN(Shape, Shape2, "");
//		DEF_INSTANCE_IN(Shape, Shape3, "");
//		DEF_INSTANCE_IN(Shape, Shape4, "");
//		DEF_INSTANCE_IN(Shape, Shape5, "");
//		DEF_INSTANCE_IN(Shape, Shape6, "");
//		DEF_INSTANCE_IN(Shape, Shape7, "");
//		DEF_INSTANCE_IN(Shape, Shape8, "");
//		DEF_INSTANCE_IN(Shape, Shape9, "");
//
//		DEF_INSTANCE_OUT(ShapeGroup, ShapeGroup, "");
//
//	protected:
//
//
//	private:
//		std::vector<FInstance<Shape>*> ShapeInstances;
//	};
//
//
//	template<typename TDataType>
//	class TextureMeshCorrect : public ComputeModule
//	{
//		DECLARE_TCLASS(TextureMeshCorrect, TDataType)
//	public:
//		typedef typename TDataType::Real Real;
//		typedef typename TDataType::Coord Coord;
//		typedef typename ::dyno::Transform<Real, 3> Transform;
//
//		TextureMeshCorrect();
//		~TextureMeshCorrect() {};
//
//		void IndexChange();
//		void onFieldChanged();
//		void compute() override { onFieldChanged(); };
//		std::string caption() override { return "TextureMeshCorrect"; }
//
//		DEF_INSTANCE_IN(TextureMesh, TextureMesh, "");
//		//Texture
//		DEF_VAR(uint,ShapeIndex,0,"");
//
//		DEF_ARRAY2D_IN(Vec4f, ColorTextureOverride, DeviceType::GPU, "ColorTextureOverride");
//		DEF_ARRAY2D_IN(Vec4f, NormalTextureOverride, DeviceType::GPU, "NormalTextureOverride");
//		DEF_ARRAY2D_IN(Vec4f, ORMTextureOverride, DeviceType::GPU, "ORMTextureOverride");
//		DEF_ARRAY2D_IN(Vec4f, AlphaTextureOverride, DeviceType::GPU, "AlphaTextureOverride");
//
//		//Color Correct
//		DEF_VAR(float, Saturation, 1, "");
//		DEF_VAR(Vec3f, Offset, Vec3f(0), "");
//		DEF_VAR(float, Contrast, 1, "");
//		DEF_VAR(float, Gamma, 1, "");
//
//		//Bump
//		DEF_VAR(float, BumpScale, 0, "");
//
//		//ORM
//		DEF_VAR(float, Metallic, 0, "");
//		DEF_VAR(float, Roughness, 0, "");
//		DEF_VAR(float, Alpha, 1, "");
//		DEF_VAR(float, AO, 1, "");
//
//		DEF_INSTANCE_OUT(TextureMesh, TextureMesh, "");
//
//	protected:
//
//	};
//
//


	/*template<typename TDataType>
	class TempUpdate : public ComputeModule
	{
		DECLARE_TCLASS(TempUpdate, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::Transform<Real, 3> Transform;

		TempUpdate() 
		{
			auto updateButton = std::make_shared<FCallBackFunc>(std::bind(&TempUpdate<TDataType>::onFieldChanged, this));
			this->varUpdate()->attach(updateButton);
		}
		~TempUpdate() {};

		void onFieldChanged();
		void compute() override { };
		std::string caption() override { return "TempUpdateButton"; }
		
		void setPipeline(std::shared_ptr<MaterialPipeline>pipeline) { matPipeline = pipeline; }

		DEF_VAR(bool, Update, false, "");

		DEF_ARRAY2D_OUT(Vec4f, Texture, DeviceType::GPU, "");

		std::shared_ptr<MaterialPipeline> matPipeline = NULL;
	protected:

	};*/



}