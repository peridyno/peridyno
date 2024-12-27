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
#include "RigidBodySystem.h"

#include "Node/ParametricModel.h"

#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

#include "FilePath.h"

namespace dyno 
{
	template<typename TDataType>
	class ArticulatedBody : virtual public ParametricModel<TDataType>, virtual public RigidBodySystem<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename Pair<uint, uint> BindingPair;

		ArticulatedBody();
		~ArticulatedBody() override;

		void bind(std::shared_ptr<PdActor> actor, Pair<uint, uint> shapeId);

	public:
		DEF_VAR(FilePath, FilePath, "", "");

		/**
		 * @brief Creates multiple vehicles and specifies the transformations for each vehicle
		 */
		DEF_VAR(std::vector<Transform3f>, VehiclesTransform, std::vector<Transform3f>{Transform3f()}, "");

		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "Texture mesh of the vechicle");

	public:
		DEF_ARRAYLIST_STATE(Transform3f, InstanceTransform, DeviceType::GPU, "Instance transforms");

		DEF_ARRAY_STATE(BindingPair, BindingPair, DeviceType::GPU, "");

		DEF_ARRAY_STATE(int, BindingTag, DeviceType::GPU, "");

		DEF_ARRAY_STATE(Matrix, InitialRotation, DeviceType::GPU, "");

	protected:
		void resetStates() override;

		void updateStates() override;

		void updateInstanceTransform();

		void clearVechicle();

		void transform();

		void varChanged();

// 		virtual std::shared_ptr<TextureMesh> getTexMeshPtr() 
// 		{
// 			if (this->inTextureMesh()->isEmpty())
// 				return NULL;
// 			else
// 				return this->inTextureMesh()->constDataPtr();
// 		};

	protected:

		DArray<Matrix> mInitialRot;

	private:
		std::vector<Pair<uint, uint>> mBindingPair;

		std::vector<std::shared_ptr<PdActor>> mActors;
	};


}
