/**
 * Copyright 2025 Lixin Ren
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
#include "AdaptiveGridGenerator.h"


namespace dyno 
{

	template<typename TDataType>
	class MSTsGeneratorDynamicUpdate : public AdaptiveGridGenerator<TDataType>
	{
		DECLARE_TCLASS(MSTsGeneratorDynamicUpdate, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		MSTsGeneratorDynamicUpdate();
		~MSTsGeneratorDynamicUpdate() override;

		void compute() override;

	private:
		void updateSeeds();

		bool mDynamic = false;

		DArray<OcKey> m_seedOld, m_seedIncrease, m_seedDecrease;
	};
}
