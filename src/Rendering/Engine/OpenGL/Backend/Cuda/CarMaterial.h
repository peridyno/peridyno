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

#include "MaterialManager.h"
#include "GraphicsObject/GLTextureMesh.h"

namespace dyno
{

	class CarMaterial : public Material
	{
	public:

		CarMaterial() {};
		~CarMaterial() override
		{
			texColor.clear();
			texBump.clear();
			texORM.clear();
			texAlpha.clear();
			texEmissive.clear();
		};

		DArray2D<Vec4f> texLightMask;

	};

	class GLCarMaterial : public GLMaterial
	{
	public:

		GLCarMaterial();
		~GLCarMaterial() override;
		void create() override;
		void release() override;

		virtual void updateGL() override;

	public:

		// color texture
		XTexture2D<dyno::Vec4f> texLightMask;

		bool mInitialized = false;
	};

	class CustomCarMaterial : public CustomMaterial
	{
	public:
		DECLARE_CLASS(CustomCarMaterial)
		MATERIAL_MANAGER_MANAGED_CLASS
		CustomCarMaterial();
		CustomCarMaterial(const std::string& name);
		CustomCarMaterial(const std::shared_ptr<MaterialLoaderModule>& MaterialLoaderPtr, std::shared_ptr<BreakMaterial>& BreakMaterialModule, std::string Name);

		DEF_ARRAY2D_IN(Vec4f, TexLightMask, DeviceType::GPU, "R:HeadLightMask; G:BrakeLightMask; B:TurnLightMask");

		virtual void updateImpl()override;

		virtual void initialVar()override;

	private:
		std::shared_ptr<CarMaterial> mCarMaterial = NULL;

	};

};