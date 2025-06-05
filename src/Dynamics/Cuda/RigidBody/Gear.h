/**
 * Copyright 2024 Yuzhong Guo
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
#include "ArticulatedBody.h"

namespace dyno
{
	template<typename TDataType>
	class Gear : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(Gear, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Gear();
		~Gear() override;

	protected:
		void resetStates() override;
	};

	template<typename TDataType>
	class Bug : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(Bug, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Bug();
		~Bug() override;
		void loadMa(std::string file_path);
		std::vector<Vec4f> vertices;
		std::vector<Vec2i> edges;
		std::vector<Vec3i>	faces;
		Vec3f position;
		Vec3f velocity;
		std::string file_name;
		void setposition(Vec3f position) { this->position = position; }
		void setv(Vec3f velocity) { this->velocity = velocity; }
		void setfile(std::string file_name) { this->file_name = file_name; }

	protected:
		void resetStates() override;
	};


}

