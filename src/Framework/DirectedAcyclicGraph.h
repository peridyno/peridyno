/**
 * Copyright 2021 Xiaowei He
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
#include "Object.h"

#include <map>
#include <list>
#include <stack>
#include <set>
#include <vector>
#include <queue>
#include <unordered_set>

namespace dyno {
	/**
	 * @brief Graph class represents a directed graph
	 *
	 */
	class DirectedAcyclicGraph
	{
	public:
		DirectedAcyclicGraph() {};
		~DirectedAcyclicGraph();

		// Add an edge to DAG
		void addEdge(ObjectId v, ObjectId w);

		// Depth first traversal of the vertices
		std::vector<ObjectId>& topologicalSort(ObjectId v);

		// Depth first traversal of the vertices
		std::vector<ObjectId>& topologicalSort();

		size_t sizeOfVertex() const;

		size_t OtherVerticesSize() { return OtherVertices.size(); }

		std::set<ObjectId>& getOtherVertices() { return OtherVertices;}

		std::set<ObjectId>& vertices();

		std::map<ObjectId, std::unordered_set<ObjectId>>& edges();

		std::map<ObjectId, std::unordered_set<ObjectId>>& reverseEdges();

		void addOtherVertices(ObjectId v);

		void addtoRemoveList(ObjectId v);

		void removeID(ObjectId v = -1, ObjectId w = -1);

	private:
		// Functions used by topologicalSort
		void topologicalSortUtil(ObjectId v, std::map<ObjectId, bool>& visited, std::stack<ObjectId>& stack);
		void topologicalSortUtil(ObjectId v, std::map<ObjectId, bool>& visited);

	private:
		std::vector<ObjectId> mOrderVertices;
		std::set<ObjectId> mVertices;
		std::map<ObjectId, std::unordered_set<ObjectId>> mEdges;
		std::map<ObjectId, std::unordered_set<ObjectId>> mReverseEdges;
		std::set<ObjectId> OtherVertices;
		std::set<ObjectId> RemoveList;
	};
}