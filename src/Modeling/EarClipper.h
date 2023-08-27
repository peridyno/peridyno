/**
 * Copyright 2023 Yuzhong Guo
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
#include "Node/ParametricModel.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	class EarClipper : public Node
	{

		DECLARE_TCLASS(EarClipper, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		void varChanged();

	public:

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_IN(PointSet<TDataType>, PointSet, "");

		DEF_VAR(bool, ReverseNormal, false, "ReverseNormal");



	protected:
		void resetStates() override;

	public:
		EarClipper();

		EarClipper(std::vector<DataType3f::Coord> vts, std::vector<TopologyModule::Triangle>& outTriangles) { polyClip(vts,outTriangles); };

		~EarClipper() { };

		void polyClip(std::vector<DataType3f::Coord> vts, std::vector<TopologyModule::Triangle>& outTriangles);

		void polyClip(DArray<Coord> vts, std::vector<TopologyModule::Triangle>& outTriangles)
		{ 
			if (vts.isEmpty()) { return; }
			CArray<Coord> c_vts;
			c_vts.assign(vts);
			
			std::vector<DataType3f::Coord> vtsvector;
			for (size_t i = 0; i < c_vts.size(); i++)
			{
				vtsvector.push_back(c_vts[i]);
			}

			this->polyClip(vtsvector, outTriangles);
		}

	};

	IMPLEMENT_TCLASS(EarClipper, TDataType);
}
