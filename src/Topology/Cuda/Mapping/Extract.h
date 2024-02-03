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
#include "Module/TopologyMapping.h"

#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"

namespace dyno
{
	template<typename TDataType>
	class ExtractEdgeSetFromPolygonSet : public TopologyMapping
	{
		DECLARE_TCLASS(ExtractEdgeSetFromPolygonSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ExtractEdgeSetFromPolygonSet();
		~ExtractEdgeSetFromPolygonSet() override;

		inline std::string caption() override { return "Extract"; }

	public:
		DEF_INSTANCE_IN(PolygonSet<TDataType>, PolygonSet, "The input polygon set");

		DEF_INSTANCE_OUT(EdgeSet<TDataType>, EdgeSet, "The output EdgeSet");

	protected:
		bool apply() override;
	};

	template<typename TDataType>
	class ExtractTriangleSetFromPolygonSet : public TopologyMapping
	{
		DECLARE_TCLASS(ExtractTriangleSetFromPolygonSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		ExtractTriangleSetFromPolygonSet();
		~ExtractTriangleSetFromPolygonSet() override;

		inline std::string caption() override { return "Extract"; }

	public:
		DEF_INSTANCE_IN(PolygonSet<TDataType>, PolygonSet, "The input polygon set");

		DEF_INSTANCE_OUT(TriangleSet<TDataType>, TriangleSet, "The output TriangleSet");

	protected:
		bool apply() override;
	};
}