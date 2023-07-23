/**
 * Copyright 2017-2022 Yuantian Cai
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
#include "EdgeSet.h"

namespace dyno
{
	class QKey
	{
	public:
		DYN_FUNC QKey()
		{
			id[0] = EMPTY;
			id[1] = EMPTY;
			id[2] = EMPTY;
			id[3] = EMPTY;
		}

		DYN_FUNC QKey(PointType v0, PointType v1, PointType v2, PointType v3)
		{
			id[0] = v0;
			id[1] = v1;
			id[2] = v2;
			id[3] = v3;


		}

		DYN_FUNC inline PointType operator[] (unsigned int i) { return id[i]; }
		DYN_FUNC inline PointType operator[] (unsigned int i) const { return id[i]; }

		DYN_FUNC inline bool operator>= (const QKey& other) const {
			if (id[0] >= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] >= other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] >= other.id[2]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] >= other.id[3]) return true;

			return false;
		}

		DYN_FUNC inline bool operator> (const QKey& other) const {
			if (id[0] > other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] > other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] > other.id[2]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] > other.id[3]) return true;

			return false;
		}

		DYN_FUNC inline bool operator<= (const QKey& other) const {
			if (id[0] <= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] <= other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] <= other.id[2]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] <= other.id[3]) return true;
			return false;
		}

		DYN_FUNC inline bool operator< (const QKey& other) const {
			if (id[0] < other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] < other.id[1]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] < other.id[2]) return true;
			if (id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] < other.id[3]) return true;
			return false;
		}

		DYN_FUNC inline bool operator== (const QKey& other) const {
			return id[0] == other.id[0] && id[1] == other.id[1] && id[2] == other.id[2] && id[3] == other.id[3];
		}

		DYN_FUNC inline bool operator!= (const QKey& other) const {
			return id[0] != other.id[0] || id[1] != other.id[1] || id[2] != other.id[2] || id[3] != other.id[3];
		}

	private:
		DYN_FUNC inline void swap(PointType& v0, PointType& v1)
		{
			PointType vt = v0;
			v0 = v0 < v1 ? v0 : v1;
			v1 = vt < v1 ? v1 : vt;
		}

		PointType id[4];
	};

	template<typename TDataType>
	class QuadSet : public EdgeSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Quad Quad;

		QuadSet();
		~QuadSet();

		DArray<Quad>& getQuads() { return mQuads; }
		void setQuads(std::vector<Quad>& quads);

		DArrayList<int>& getVertex2Quads();

		//void loadObjFile(std::string filename);

		void copyFrom(QuadSet<TDataType>& quadSet);
		
		bool isEmpty() override;

	public:
		DEF_ARRAY_OUT(Coord, VertexNormal, DeviceType::GPU, "");

	protected:
		void updateTopology() override;

		void updateEdges() override;

		void updateVertexNormal();

		virtual void updateQuads() {};

	private:
		DArray<Quad> mQuads;
		DArrayList<int> mVer2Quad;

		DArray<::dyno::TopologyModule::Edg2Quad> mEdg2Quad;
	};
}

