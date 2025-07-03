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
	class MatBody : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(MatBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MatBody();
		~MatBody() override;
		void loadMa(std::string file_path, int objectId);
		void setXMLPath(std::string file_path)
		{
			this->mXmlPath = file_path;
		}

		std::vector<std::vector<Vec4f>> Vertices;
		std::vector<std::vector<Vec2i>> Edges;
		std::vector<std::vector<Vec3i>>	Faces;

	protected:
		void resetStates() override;

	private:
		std::string mXmlPath;
	};


}

