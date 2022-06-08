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
#include "DirectedAcyclicGraph.h"

namespace dyno {
	/**
	 * @brief Automatic layout for directed acyclic graph
	 *			Refer to "Sugiyama Algorithm" by Nikola S. Nikolov[2015] for details
	 */
	class AutoLayoutDAG
	{
	public:
		AutoLayoutDAG(DirectedAcyclicGraph* dag);
		~AutoLayoutDAG();

		void update();

		size_t layerNumber() { return mLayerNum; }

		std::vector<ObjectId>& layer(size_t l) { return mNodeLayers[l]; }

	protected:
		void constructHierarchy();

		void addDummyVertices();

		void minimizeEdgeCrossings();

	private:
		DirectedAcyclicGraph* pDAG;

		std::set<ObjectId> mVertices;
		std::map<ObjectId, std::unordered_set<ObjectId>> mEdges;
		std::map<ObjectId, std::unordered_set<ObjectId>> mReverseEdges;

		std::map<ObjectId, int> mLayers;
		std::map<ObjectId, int> mXCoordinate;

		std::vector<std::vector<ObjectId>> mNodeLayers;

		size_t mLayerNum = 0;

		int mIterNum = 1;
	};
}
