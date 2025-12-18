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
#include "TriangleSet.h"
#include <set>
#include <regex>
#include "Module/Pipeline.h"
#include "Field/Color.h"
#include "Module/ComputeModule.h"


#define MATERIAL_MANAGER_MANAGED_CLASS friend class MaterialManager;

namespace dyno
{
	class MaterialManager;
	class MaterialPipeline;
	class MaterialManagerObserver;


	class MaterialManagedModule : public Module
	{
		MATERIAL_MANAGER_MANAGED_CLASS
	public:
		MaterialManagedModule() :Module() {};
		std::string getName() const { return mName; }

	protected:
		void setName(std::string name) { mName = name; }
		std::string mName = "default";
		virtual std::shared_ptr<MaterialManagedModule> clone() const 
		{
			std::cout << mName << " : Need clone()!\n";
			return nullptr;
		};
	};

	class Material : public MaterialManagedModule
	{
	public:
		DECLARE_CLASS(Material)
		MATERIAL_MANAGER_MANAGED_CLASS

		Material():MaterialManagedModule()
		{
			initial();
		};
		~Material() override;

		virtual void initial();

		void updateVar2Out();

		virtual void updateImpl()override;

		DEF_VAR(Color, BaseColor,Color(0.8), "");
		DEF_VAR(float, Metallic,0, "");
		DEF_VAR(float, Roughness,0.5, "");
		DEF_VAR(float, Alpha,1, "");
		DEF_VAR(float, BumpScale,1, "");

		DEF_VAR_OUT(Vec3f,BaseColor,"");
		DEF_VAR_OUT(float, Metallic, "");
		DEF_VAR_OUT(float, Roughness, "");
		DEF_VAR_OUT(float, Alpha, "");
		DEF_VAR_OUT(float, BumpScale, "");

		DEF_ARRAY2D_OUT(Vec4f,TexColor,DeviceType::GPU,"");
		DEF_ARRAY2D_OUT(Vec4f, TexBump, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexORM, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexAlpha, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexEmissive, DeviceType::GPU, "");

	public:
		
		void addAssigner(std::shared_ptr<Module> assigner);

		void removeAssigner(std::shared_ptr<Module> assigner);

	protected:
		void updateAssigner();

		std::shared_ptr<MaterialManagedModule> clone() const;


	private:

		Material(const std::string& name);
		Material(std::shared_ptr<Material> other);
		std::set<std::shared_ptr<Module>> mAssigner;
	};

	class CustomMaterial : public Material
	{
	public:
		DECLARE_CLASS(CustomMaterial)
		MATERIAL_MANAGER_MANAGED_CLASS
		CustomMaterial():Material() {}
		CustomMaterial(const std::string& name) :Material() { this->mName = name; }

		std::shared_ptr<Material> getBaseMaterial() const {
			return mBaseMaterial;
		}

		DEF_VAR_IN(Vec3f, BaseColor, "");
		DEF_VAR_IN(float, Metallic, "");
		DEF_VAR_IN(float, Roughness, "");
		DEF_VAR_IN(float, Alpha, "");
		DEF_VAR_IN(float, BumpScale, "");

		DEF_ARRAY2D_IN(Vec4f, TexColor, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexBump, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexORM, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexAlpha, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexEmissiveColor, DeviceType::GPU, "");

		void initial()override;

	public:

		virtual void updateImpl()override;

		std::shared_ptr<MaterialPipeline> materialPipeline();
		std::string pushMaterialManagedModule(std::shared_ptr<MaterialManagedModule> managedModule);
		std::vector<std::shared_ptr<Module>> piplineModules();

	private:

		CustomMaterial(std::shared_ptr<Material> sourceMaterial, const std::string& name = std::string("Material"));
		std::shared_ptr<Material> mBaseMaterial;

		std::shared_ptr<MaterialPipeline>	mMaterialPipeline = NULL;
	};

	class MaterialManager {
	public:
		MaterialManager() = delete;
		~MaterialManager() = delete;



		static std::shared_ptr<Material> NewMaterial();

		static std::shared_ptr<Material> NewMaterial(std::string name);

		static std::shared_ptr<Material> NewMaterial(const Material& other);

		static std::shared_ptr<CustomMaterial> createCustomMaterial(std::shared_ptr<Material> sourceMaterial = NULL, std::string name = "Material");

		void rename(std::shared_ptr<Material> ptr, const std::string& name);

		static std::shared_ptr<MaterialManagedModule> copyMaterialManagedModule(std::shared_ptr<MaterialManagedModule> material);

		static std::shared_ptr<Material> getMaterial(const std::string& name);

		static std::shared_ptr<MaterialManagedModule> getMaterialManagedModule(const std::string& name);

		static bool removeMaterialManagedModule(const std::string& name);

		static void clear() { materials().clear(); }

		static void printAllMaterials();
		static void printAllManagedModules();

		static std::string generateUniqueMaterialName(const std::string& baseName);

		static std::map<std::string, std::shared_ptr<Material>>& materials();

		static bool containsMaterial(const std::shared_ptr<Material>& mat);

		static bool containsModule(const std::shared_ptr<MaterialManagedModule>& matModule);

		static std::map<std::string, std::shared_ptr<MaterialManagedModule>>& materialManagedModules();

		static std::map<std::string, int>& nameCount();

		static void addObserver(MaterialManagerObserver* observer);

		static void removeObserver(MaterialManagerObserver* observer);

		static void callMaterialManagerObservers(std::shared_ptr<Material> mat = NULL);

		static std::string pushMaterialManagedModule(std::shared_ptr<MaterialManagedModule> managedModule,bool checkName = true);

	private:
		static std::vector<MaterialManagerObserver*> mObservers;
		static std::string addMaterial(std::shared_ptr<Material> material);
	};


	class MaterialPipeline :public Pipeline
	{
	public:
		MaterialPipeline(std::shared_ptr<CustomMaterial> CustomMaterial) : Pipeline(nullptr)
		{
			mCustomMaterial = CustomMaterial;
		}

		virtual void pushModule(std::shared_ptr<Module> m)override;

		virtual void popModule(std::shared_ptr<Module> m)override;

		void updateMaterialPipline();

	protected:
		void reconstructPipeline()override;
		std::shared_ptr<CustomMaterial> mCustomMaterial;
	};

	class MaterialManagerObserver {
	public:
		virtual ~MaterialManagerObserver() = default;
		virtual void onMaterialChanged(std::shared_ptr<MaterialManagedModule> mat) = 0;
	};

};