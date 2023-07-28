/**
 * Copyright 2017-2023 Xiaowei He
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
#include "PointSet.h"

namespace dyno
{
	class EKey
	{
	public:
		DYN_FUNC EKey()
		{
			id[0] = EMPTY;
			id[1] = EMPTY;
		}

		DYN_FUNC EKey(PointType v0, PointType v1)
		{
			id[0] = v0;
			id[1] = v1;

			swap(id[0], id[1]);
		}

		DYN_FUNC inline PointType operator[] (unsigned int i) { return id[i]; }
		DYN_FUNC inline PointType operator[] (unsigned int i) const { return id[i]; }

		DYN_FUNC inline bool operator>= (const EKey& other) const {
			if (id[0] >= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] >= other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator> (const EKey& other) const {
			if (id[0] > other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] > other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator<= (const EKey& other) const {
			if (id[0] <= other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] <= other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator< (const EKey& other) const {
			if (id[0] < other.id[0]) return true;
			if (id[0] == other.id[0] && id[1] < other.id[1]) return true;

			return false;
		}

		DYN_FUNC inline bool operator== (const EKey& other) const {
			return id[0] == other.id[0] && id[1] == other.id[1];
		}

		DYN_FUNC inline bool operator!= (const EKey& other) const {
			return id[0] != other.id[0] || id[1] != other.id[1];
		}

		DYN_FUNC inline bool isValid() const {
			return id[0] != EMPTY && id[1] != EMPTY;
		}

	private:
		DYN_FUNC inline void swap(PointType& v0, PointType& v1)
		{
			PointType vt = v0;
			v0 = v0 < v1 ? v0 : v1;
			v1 = vt < v1 ? v1 : vt;
		}

		PointType id[2];
	};

	template<typename TDataType>
	class EdgeSet : public PointSet<TDataType>
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TopologyModule::Edge Edge;

		EdgeSet();
		~EdgeSet() override;

		/**
		 * @brief Request the neighboring ids of each point according to the mesh topology
		 * 			Be sure update() is called as long as the topology is changed  
		 * 
		 * @param lists A neighbor list storing the ids
		 */
		void requestPointNeighbors(DArrayList<int>& lists);

		void loadSmeshFile(std::string filename);

		/**
		 * @brief Get all edges with each one containing the indices of two edge ends
		 * 
		 * @return DArray<Edge>& A GPU array
		 */
		DArray<Edge>& getEdges() {return mEdges;}

		/**
		 * @brief Get the Ver2 Edge object
		 * 
		 * @return DArrayList<int>& 
		 */
		DArrayList<int>& getVer2Edge();

		void setEdges(std::vector<Edge>& edges);
		void setEdges(DArray<Edge>& edges);

		void copyFrom(EdgeSet<TDataType>& edgeSet);

		bool isEmpty() override;

	protected:
		/**
		 * Override updateEdges to update edges in a customized way, e.g., only the four edges will be created for a quadrangle
		 */
		virtual void updateEdges() {};

		void updateTopology() override;

	protected:
		DArray<Edge> mEdges;
		DArrayList<int> mVer2Edge;
	};
}

