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

	template<typename TDataType>
	class FloatingNumber : public DataSource
	{
		DECLARE_TCLASS(FloatingNumber, TDataType);
	public:
		typedef typename TDataType::Real Real;

		FloatingNumber();

		DEF_VAR(Real, Value, Real(0), "Initial value");

		DEF_VAR_OUT(Real, Floating, "");
	};

	template<typename TDataType>
	class Vector3Source : public DataSource
	{
		DECLARE_TCLASS(Vector3Source, TDataType);
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Vector3Source();

		DEF_VAR(Coord, Value, Real(0), "Initial value");

		DEF_VAR_OUT(Coord, Vector, "");
	};
}
