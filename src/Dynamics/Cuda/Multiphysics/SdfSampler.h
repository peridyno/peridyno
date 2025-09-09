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

#include <Volume/Volume.h >
#include <Samplers/Sampler.h>

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

		DEF_VAR(float, Spacing, Real(0.02), " ");
	public:
		DEF_NODE_PORT(Volume<TDataType>, Volume, "");

	protected:
		void resetStates() override;

		bool validateInputs() override;

	private:
		std::shared_ptr<DistanceField3D<TDataType>> m_inputSDF;

		Coord mCubeTilt = Coord(0.0f);

		Coord mX = Coord(1.0f, 0.0f, 0.0f);	//Cube Init X Rotation
		Coord mY = Coord(0.0f, 1.0f, 0.0f);	//Cube Init Y Rotation
		Coord mZ = Coord(0.0f, 0.0f, 1.0f);	//Cube Init Z Rotation

		Real mAlpha = 0.0f;
		Real mBeta = 0.0f;
		Real mGamma = 0.0f;
	};
}
