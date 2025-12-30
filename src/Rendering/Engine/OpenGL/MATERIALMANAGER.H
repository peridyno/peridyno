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
#include "Topology/TriangleSet.h"
#include <set>
#include <regex>
#include "Module/Pipeline.h"
#include "Field/Color.h"
#include "Module/KeyboardInputModule.h"
#include "Module/MouseInputModule.h"
#include "Node.h"
#include "Module/ComputeModule.h"
#include "Topology/TextureMesh.h"


#define MATERIAL_MANAGER_MANAGED_CLASS friend class MaterialManager;

namespace dyno
{
	class MaterialManager;
	class MaterialPipeline;
	class MaterialManagerObserver;
	class CustomMaterial;

	class MaterialAction
	{
	public:
		MaterialAction() {};
		virtual ~MaterialAction() {};

		virtual void start(std::shared_ptr<CustomMaterial> customMaterial) {};
		virtual void process(std::shared_ptr<CustomMaterial> customMaterial){};
		virtual void end(std::shared_ptr<CustomMaterial> customMaterial) {};
	private:

	};


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

	class MaterialLoaderModule : public MaterialManagedModule
	{
	public:
		DECLARE_CLASS(MaterialLoaderModule)
		MATERIAL_MANAGER_MANAGED_CLASS

		MaterialLoaderModule():MaterialManagedModule()
		{
			initialVar();
			this->outMaterial()->setDataPtr(std::make_shared<Material>());
		};
		MaterialLoaderModule(std::shared_ptr<Material> sourceMaterial, std::string name) 
		{
			this->outMaterial()->setDataPtr(sourceMaterial); 
			this->setName(name);
		}
		~MaterialLoaderModule() override {};

		void initialVar() 
		{
			this->varAlpha()->setRange(0, 1);
			this->varBumpScale()->setRange(0, 1);
			this->varEmissiveIntensity()->setRange(0, 1);
			this->varMetallic()->setRange(0, 1);
			this->varRoughness()->setRange(0, 1);
		}

		//DEF_VAR(std::string,Name,"", "");
		DEF_VAR(Color, BaseColor,Color(0.8), "");
		DEF_VAR(float, Metallic,0, "");
		DEF_VAR(float, Roughness,0.5, "");
		DEF_VAR(float, Alpha,1, "");
		DEF_VAR(float, BumpScale,1, "");
		DEF_VAR(float, EmissiveIntensity, 0, "");

		DEF_INSTANCE_OUT(Material,Material, "");


	private:

		MaterialLoaderModule(const std::string& name);
		MaterialLoaderModule(std::shared_ptr<MaterialLoaderModule> other);

	};

	class BreakMaterial : public MaterialManagedModule
	{
		DECLARE_CLASS(BreakMaterial)

	public:

		BreakMaterial() { initial(); };
		BreakMaterial(std::shared_ptr<BreakMaterial> other) ;
		~BreakMaterial() {}

		void onFieldChanged();
		void updateImpl() override { onFieldChanged(); };
		std::string caption() override { return "BreakMaterial"; }
		void initial();
		//Texture
		DEF_INSTANCE_IN(Material, Material, "Material");

		DEF_VAR_OUT(Color, Color, "");
		DEF_VAR_OUT(float, Roughness, "");
		DEF_VAR_OUT(float, Metallic, "");
		DEF_VAR_OUT(float, Alpha, "");
		DEF_VAR_OUT(float, BumpScale, "");
		DEF_VAR_OUT(float, EmissiveIntensity, "");

		DEF_ARRAY2D_OUT(Vec4f, TexColor, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexBump, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexORM, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexAlpha, DeviceType::GPU, "");
		DEF_ARRAY2D_OUT(Vec4f, TexEmissive, DeviceType::GPU, "");

	protected:
		virtual std::shared_ptr<MaterialManagedModule> clone() const override;

	};

	class CustomMaterial : public MaterialManagedModule
	{
	public:
		DECLARE_CLASS(CustomMaterial)
		MATERIAL_MANAGER_MANAGED_CLASS
		CustomMaterial();
		CustomMaterial(const std::string& name);
		CustomMaterial(const std::shared_ptr<MaterialLoaderModule>& MaterialLoaderPtr, std::shared_ptr<BreakMaterial>& BreakMaterialModule, std::string Name);

		virtual void updateImpl();
		void updateVar2Out();
		DEF_VAR(Color, BaseColor, Color(0.8), "");
		DEF_VAR(float, Metallic, 0, "");
		DEF_VAR(float, Roughness, 0.5, "");
		DEF_VAR(float, Alpha, 1, "");
		DEF_VAR(float, BumpScale, 1, "");
		DEF_VAR(float, EmissiveIntensity, 0, "");

		DEF_VAR_IN(Color, BaseColor, "");
		DEF_VAR_IN(float, Metallic, "");
		DEF_VAR_IN(float, Roughness, "");
		DEF_VAR_IN(float, Alpha, "");
		DEF_VAR_IN(float, BumpScale, "");
		DEF_VAR_IN(float, EmissiveIntensity, "");

		DEF_ARRAY2D_IN(Vec4f, TexColor, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexBump, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexORM, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexAlpha, DeviceType::GPU, "");
		DEF_ARRAY2D_IN(Vec4f, TexEmissiveColor, DeviceType::GPU, "");

		DEF_INSTANCE_OUT(Material,Material,"");

		void initialVar()
		{
			this->varAlpha()->setRange(0, 1);
			this->varBumpScale()->setRange(0, 1);
			this->varEmissiveIntensity()->setRange(0, 1);
			this->varMetallic()->setRange(0, 1);
			this->varRoughness()->setRange(0, 1);

			this->inAlpha()->tagOptional(true);
			this->inBaseColor()->tagOptional(true);
			this->inBumpScale()->tagOptional(true);
			this->inEmissiveIntensity()->tagOptional(true);
			this->inRoughness()->tagOptional(true);
			this->inMetallic()->tagOptional(true);

			this->inTexAlpha()->tagOptional(true);
			this->inTexBump()->tagOptional(true);
			this->inBaseColor()->tagOptional(true);
			this->inTexEmissiveColor()->tagOptional(true);
			this->inTexORM()->tagOptional(true);
		}

		void addAssigner(std::shared_ptr<Module> assigner);
			          
		void removeAssigner(std::shared_ptr<Module> assigner);
		
		void updateAssigner();

	public:

		std::shared_ptr<MaterialPipeline> materialPipeline();
		std::string pushMaterialManagedModule(std::shared_ptr<MaterialManagedModule> managedModule);
		std::vector<std::shared_ptr<Module>> piplineModules();

	private:

		std::shared_ptr<MaterialPipeline>	mMaterialPipeline = NULL;
		std::set<std::shared_ptr<Module>> mAssigner;
	};

	class MaterialManager {
	public:
		MaterialManager() = delete;
		~MaterialManager() = delete;

		static std::shared_ptr<MaterialLoaderModule> createMaterialLoaderModule(std::shared_ptr<Material> mat, std::string Name);		

		static std::shared_ptr<CustomMaterial> createCustomMaterial(std::string name = "Material");
		
		static std::shared_ptr<CustomMaterial> createCustomMaterial(const std::shared_ptr<MaterialLoaderModule>& MaterialLoaderPtr, std::shared_ptr<BreakMaterial>& BreakMaterialModule, std::string Name = "Material");

		void rename(std::shared_ptr<MaterialManagedModule> ptr, const std::string& name);

		static std::shared_ptr<MaterialManagedModule> copyMaterialManagedModule(std::shared_ptr<MaterialManagedModule> material);

		static std::shared_ptr<CustomMaterial> getMaterial(const std::string& name);

		static std::shared_ptr<MaterialManagedModule> getMaterialManagedModule(const std::string& name);

		static bool removeMaterialManagedModule(const std::string& name);

		static void clear() { materials().clear(); }

		static void printAllMaterials();
		
		static void printAllManagedModules();

		static std::string generateUniqueMaterialName(const std::string& baseName);

		static std::map<std::string, std::shared_ptr<CustomMaterial>>& materials();

		static bool containsMaterial(const std::shared_ptr<CustomMaterial>& mat);

		static bool containsModule(const std::shared_ptr<MaterialManagedModule>& matModule);

		static std::map<std::string, std::shared_ptr<MaterialManagedModule>>& materialManagedModules();

		static std::map<std::string, int>& nameCount();

		static void addMaterialListObserver(MaterialManagerObserver* observer);

		static void removeMaterialListObserver(MaterialManagerObserver* observer);

		static void callMaterialManagerObservers(std::shared_ptr<MaterialManagedModule> mat = NULL);

		static std::string pushMaterialManagedModule(std::shared_ptr<MaterialManagedModule> managedModule,bool checkName = true);

		static void onKeyboardEvent(PKeyboardEvent event);

		static void traverseForward(MaterialAction& matAction);

	private:
		static std::vector<MaterialManagerObserver*> mMaterialListObservers;
		static std::string addMaterial(std::shared_ptr<CustomMaterial> CustomMaterial);

		static std::vector<std::shared_ptr<MaterialLoaderModule>> mMaterialLoaderModules;
	};


	class MaterialPipeline :public Pipeline
	{
	public:
		MaterialPipeline(CustomMaterial* CustomMaterial) : Pipeline(nullptr)
		{
			mCustomMaterial = CustomMaterial;
		}

		virtual void pushModule(std::shared_ptr<Module> m)override;

		virtual void popModule(std::shared_ptr<Module> m)override;

		void updateMaterialPipline();

	public:
		bool autoUpdate = false;

	protected:
		void reconstructPipeline()override;
		CustomMaterial* mCustomMaterial;
	};

	class MaterialManagerObserver {
	public:
		virtual ~MaterialManagerObserver() = default;
		virtual void onMaterialListChanged(std::shared_ptr<MaterialManagedModule> mat) = 0;
	};

	//class MaterialUpdateObserver {
	//public:
	//	virtual ~MaterialUpdateObserver() = default;
	//	virtual void onMaterialUpdate(std::shared_ptr<Material> mat) = 0;
	//};

};