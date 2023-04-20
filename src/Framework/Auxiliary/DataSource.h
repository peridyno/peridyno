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
#include "Module.h"

namespace dyno 
{
	class DataSource : public Module
	{
	public:
		DataSource() {};
		~DataSource() override {};

		bool captionVisible() override;

		std::string getModuleType() override { return "Variables"; }
	};

	//float
	class FloatSource : public DataSource
	{
		DECLARE_CLASS(FloatSource);
	public:
		FloatSource();

		DEF_VAR(float, Value, 0.0f, "Initial value");

		DEF_VAR_OUT(float, Float, "");
	};

	//double
	class DoubleSource : public DataSource
	{
		DECLARE_CLASS(DoubleSource);
	public:
		DoubleSource();

		DEF_VAR(double, Value, 0.0, "Initial value");

		DEF_VAR_OUT(double, Double, "");
	};

	//Vec3f
	class Vec3fSource : public DataSource
	{
		DECLARE_CLASS(Vec3fSource);
	public:
		Vec3fSource();

		DEF_VAR(Vec3f, Value, Vec3f(0.0f), "Initial value");

		DEF_VAR_OUT(Vec3f, Vector3f, "");
	};

	//Vec3f
	class Vec3dSource : public DataSource
	{
		DECLARE_CLASS(Vec3dSource);
	public:
		Vec3dSource();

		DEF_VAR(Vec3d, Value, Vec3d(0.0), "Initial value");

		DEF_VAR_OUT(Vec3d, Vector3d, "");
	};
}
