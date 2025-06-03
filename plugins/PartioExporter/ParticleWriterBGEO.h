/**
 * Copyright 2025 Xukun Luo
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
#include "Array/Array.h"
#include "Module/OutputModule.h"

namespace dyno
{
	template<typename TDataType>
	class ParticleWriterBGEO : public OutputModule
	{
		DECLARE_TCLASS(ParticleWriterBGEO, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleWriterBGEO();
		virtual ~ParticleWriterBGEO();

		void output() override;

	public:
		DEF_ARRAY_IN(Vec3f, Position, DeviceType::GPU, "");

		DEF_VAR(int, AttributeNum, 0, "Attribute number");
		DEF_ARRAY_IN(Real, Attribute0, DeviceType::GPU, "");	
		DEF_ARRAY_IN(Real, Attribute1, DeviceType::GPU, "");
		DEF_ARRAY_IN(Real, Attribute2, DeviceType::GPU, "");
		DEF_ARRAY_IN(Real, Attribute3, DeviceType::GPU, "");
		DEF_ARRAY_IN(Real, Attribute4, DeviceType::GPU, "");

		DEF_VAR(std::string, AttributeName0, "Attribute0", "Attribute0 name");
		DEF_VAR(std::string, AttributeName1, "Attribute0", "Attribute0 name");
		DEF_VAR(std::string, AttributeName2, "Attribute0", "Attribute0 name");
		DEF_VAR(std::string, AttributeName3, "Attribute0", "Attribute0 name");
		DEF_VAR(std::string, AttributeName4, "Attribute0", "Attribute0 name");

		DEF_VAR(int, Interval, 8.0f, "Output interval frame number");
		

	private:
		int m_output_index = 0;
		int time_idx = 0;


		CArray<Coord> m_positions;
		CArray<Real> m_attributes[5];
	};
}
