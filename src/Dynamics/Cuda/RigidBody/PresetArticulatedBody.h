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

#include "STL/Pair.h"
#include "VehicleInfo.h"
#include "FilePath.h"

namespace dyno 
{
	template<typename TDataType>
	class PresetArticulatedBody : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(PresetArticulatedBody, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PresetArticulatedBody();
		~PresetArticulatedBody() override;

		DEF_VAR(FilePath, FilePath, "", "");
		DEF_INSTANCE_STATE(TextureMesh, TextureMeshState, "Texture mesh of the vechicle");


	protected:
		void resetStates() override;

		std::shared_ptr<TextureMesh> getTexMeshPtr()override 
		{
			if (this->stateTextureMeshState()->isEmpty())
				return NULL;
			else
				return this->stateTextureMeshState()->constDataPtr();
		};

	private:
		void varChanged();

	};

	template<typename TDataType>
	class PresetJeep : virtual public PresetArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(PresetJeep, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PresetJeep();
		~PresetJeep() override;

	protected:
		void resetStates() override;


	};

	template<typename TDataType>
	class PresetTank : virtual public PresetArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(PresetJeep, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		PresetTank();
		~PresetTank() override;

	protected:
		void resetStates() override;

	};

}
