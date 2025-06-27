/**
 * Copyright 2023~2024 Hao He, Shusen Liu.
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
#include "Node.h"
#include "Topology/LevelSet.h"
#include "Topology/HexahedronSet.h"
#include <cmath>
#include <Volume/VolumeOctree.h>
#include <Volume/Volume.h >
#include "Samplers/Sampler.h"

namespace dyno {

	template<typename TDataType>
	class SdfSampler : public Sampler<TDataType>
	{
		DECLARE_TCLASS(SdfSampler, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Hexahedron Hexahedron;

		SdfSampler();
		~SdfSampler() override;

		void resetStates() override;

		bool validateInputs() override;

		std::shared_ptr<dyno::DistanceField3D<TDataType>>  convert2Uniform(
			VolumeOctree<TDataType>* volume,
			Real h);

	public:

		DEF_NODE_PORT(Volume<TDataType>, Volume, "");

		DEF_NODE_PORT(VolumeOctree<TDataType>, VolumeOctree, "");



		//DEF_INSTANCE_OUT(PointSet<TDataType>, PointSet, "");

		DEF_VAR(float, Spacing, Real(0.02), " ");

		DEF_VAR(Vec3f, CubeTilt, 0, "Cube Init Rotation");

		DEF_VAR(Vec3f, X, Coord(1, 0.0f, 0.0f), "Cube Init X Rotation");
		DEF_VAR(Vec3f, Y, Coord(0.0f, 1, 0.0f), "Cube Init Y Rotation");
		DEF_VAR(Vec3f, Z, Coord(0.0f, 0.0f, 1), "Cube Init Z Rotation");

		DEF_VAR(Real, Alpha, Real(0), " ");
		DEF_VAR(Real, Beta, Real(0), " ");
		DEF_VAR(Real, Gamma, Real(0), " ");
	};
}
