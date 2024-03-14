/**
 * Copyright 2022 Yuantian Cai
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
#include <Module.h>
#include <Module/TopologyModule.h>
#include "Topology/TetrahedronSet.h"
#include "Topology/HexahedronSet.h"
#include "Topology/QuadSet.h"
#include "Topology/GeoVisualSet.h"
namespace dyno
{
	/*!
	*	\class	ProcessGeo
	*	\brief	This class used to parse data from GeoVisualSet to basic geometry class.
	*/

	template<typename TDataType>
	class ProcessGeo : public Module
	{
		DECLARE_TCLASS(ProcessGeo, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;
		typedef typename TopologyModule::Quad Quad;
		typedef typename TopologyModule::Hexahedron Hexahedron;

		//基本显示数据
		PointSet3f node;
		HexahedronSet3f hexElement;
		QuadSet3f  quadElement;
		TriangleSet3f triElement;
		TetrahedronSet3f tetElement;

		ProcessGeo();
		~ProcessGeo() override;
		void setAllPoints();
		void setPoints();
		void setTriangles();
		void setQuads();
		void setTetrahedrons();
		void setHexahedrons();

		DEF_INSTANCE_OUT(PointSet3f, AllPointSet, "get all PointSet");
		DEF_INSTANCE_IN(GeoVisualSet3f, GeoVisualSet, "GeoVisualSet");
		DEF_INSTANCE_OUT(PointSet3f, PointSet, "PointSet");
		DEF_INSTANCE_OUT(HexahedronSet3f, HexahedronSet, "HexahedronSet");
		DEF_INSTANCE_OUT(QuadSet3f, QuadSet, "QuadSet");
		DEF_INSTANCE_OUT(TriangleSet3f, TriangleSet, "TriangleSet");
		DEF_INSTANCE_OUT(TetrahedronSet3f, TetrahedronSet, "TetrahedronSet");

		void updateImpl() override;

	};
}

