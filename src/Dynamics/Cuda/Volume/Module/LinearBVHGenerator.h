/**
 * Copyright 2024 Lixin Ren
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
	class vBVHNode
	{
	public:
		DYN_FUNC vBVHNode()
		{
			m_level = 0;
			m_morton = 0;
			parent = EMPTY;
			left = EMPTY;
			right = EMPTY;
		};

		DYN_FUNC vBVHNode(Level l, OcKey m)
		{
			m_level = l;
			m_morton = m;
			parent = EMPTY;
			left = EMPTY;
			right = EMPTY;
		};

		DYN_FUNC bool isLeaf() { return left == EMPTY && right == EMPTY; }

		Level m_level;
		OcKey m_morton;

		int parent;
		int left;
		int right;
	};

	template<typename TDataType>
	class LinearBVHGenerator : public AdaptiveGridGenerator<TDataType>
	{
		DECLARE_TCLASS(LinearBVHGenerator, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		LinearBVHGenerator() {};
		~LinearBVHGenerator() override {};

		void compute() override;

	};
}
