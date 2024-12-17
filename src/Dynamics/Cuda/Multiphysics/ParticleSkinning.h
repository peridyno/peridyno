/**
 * Copyright 2023 Shusen Liu
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

#include "ParticleSystem/ParticleSystem.h"

#include "Topology/LevelSet.h"
#include "Topology/TriangleSet.h"

namespace dyno
{
	template<typename TDataType>
	class ParticleSkinning : public Node
	{
		DECLARE_TCLASS(ParticleSkinning, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ParticleSkinning() ;
		~ParticleSkinning() override {};

	public:
		DEF_NODE_PORT(ParticleSystem<TDataType>, ParticleSystem, "Initial Fluid Particles");

		DEF_ARRAY_STATE(Coord, Points, DeviceType::GPU, "Point positions");

		DEF_INSTANCE_STATE(LevelSet<TDataType>, LevelSet, "A 3D signed distance field");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "An iso surface");

		DEF_ARRAY_STATE(Coord, GridPoistion, DeviceType::GPU, "Grid positions");

		DEF_VAR_STATE(Real, GridSpacing, 0.01, "Grid spacing");

	protected:
		void resetStates() override;

		void preUpdateStates() override;

	private:
		void updateLevelset();

		void constrGridPositionArray();
	};


}