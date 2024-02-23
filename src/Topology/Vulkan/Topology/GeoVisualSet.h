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
namespace dyno
{
	/*!
	*	\class	GeoVisualSet
	*	\brief	This class used to store general geometry data .
	*/

	template<typename TDataType>
	class GeoVisualSet : public Module
	{
		DECLARE_TCLASS(GeoVisualSet, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Triangle Triangle;
		typedef typename TopologyModule::Tetrahedron Tetrahedron;
		typedef typename TopologyModule::Quad Quad;
		typedef typename TopologyModule::Hexahedron Hexahedron;

		GeoVisualSet();
		~GeoVisualSet() override;
		void setPoints(std::vector<Coord>& points);
		void setTriangles(std::vector<Coord>& node, std::vector<Triangle>& triangles);
		void setQuads(std::vector<Coord>& node, std::vector<Quad>& quads);
		void setTetrahedrons(std::vector<Coord>& node, std::vector<Tetrahedron>& tetrahedrons);
		void setHexahedrons(std::vector<Coord>& node, std::vector<Hexahedron>& hexahedrons);

		//包含以下的所有点数组
		std::vector<Coord> allPoints;
		//点数组
		std::vector<Coord> points;
		//三角形数组
		std::vector<Coord> trinode;
		std::vector<Triangle> triangles;
		//四边形数组
		std::vector<Coord> quadnode;
		std::vector<Quad> quads;
		//四面体数组
		std::vector<Coord> tetnode;
		std::vector<Quad> tetrahedrons;
		//六面体
		std::vector<Coord> hexnode;
		std::vector<Hexahedron> hexahedrons;
		
	};

	using GeoVisualSet3f = GeoVisualSet<DataType3f>;
}

