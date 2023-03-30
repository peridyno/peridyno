/**
 * Copyright 2017-2021 Xiaowei He
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
#include "ParticleSystem/ParticleSystem.h"

namespace dyno
{
	template<typename TDataType>
	class GLPointVisualNode : public Node
	{
		DECLARE_TCLASS(GLPointVisualNode, TDataType)
	public:
		typedef typename TDataType::Coord Coord;

		GLPointVisualNode();
		~GLPointVisualNode() override;

	public:
		void resetStates() override;

		void preUpdateStates() override;

		DEF_NODE_PORT(ParticleSystem<TDataType>, Particles, "Particles");

		DEF_VAR_IN(Real, Test, "");

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSetIn, "");

		DEF_INSTANCE_OUT(PointSet<TDataType>, PointSetOut, "");

	public:
		DEF_ARRAY_STATE(Coord, Vector, DeviceType::GPU, "");

		DEF_INSTANCE_STATE(TopologyModule, Topology, "Topology");

	};
};
