#include "AutoLayoutDAG.h"

namespace dyno 
{
	AutoLayoutDAG::AutoLayoutDAG(DirectedAcyclicGraph* dag)
	{
		pDAG = dag;

		auto& vertices = pDAG->vertices();
		auto& edges = pDAG->edges();

		for each (auto v in vertices)
		{
			mVertices.insert(v);
		}

		for each (auto it in edges)
		{
			ObjectId v = it.first;
			for each (auto w in it.second)
			{
				// Add an edge (v, w).
				mEdges[v].insert(w);

				//Add a reverse edge (w, v)
				mReverseEdges[w].insert(v);
			}
		}
	}

	AutoLayoutDAG::~AutoLayoutDAG()
	{
		for each (auto it in mEdges)
		{
			it.second.clear();
		}
		mEdges.clear();

		for each (auto it in mReverseEdges)
		{
			it.second.clear();
		}
		mReverseEdges.clear();

		mLayers.clear();
		mXCoordinate.clear();

		for (size_t i = 0; i < mNodeLayers.size(); i++)
		{
			mNodeLayers[i].clear();
		}
		mNodeLayers.clear();
	}

	void AutoLayoutDAG::update()
	{
		constructHierarchy();

		addDummyVertices();

		minimizeEdgeCrossings();
	}

	void AutoLayoutDAG::constructHierarchy()
	{
		int maxLayer = mVertices.size();

		std::map<ObjectId, bool> visited;
		std::map<ObjectId, bool> isActive;

		std::queue<ObjectId> activeId;

		//Push vertices that have not outgoing edges into a queue
		for each (auto v in mVertices)
		{
			if (mEdges[v].size() == 0) {
				mLayers[v] = maxLayer;

				activeId.push(v);
				isActive[v] = true;
			}
			else {
				mLayers[v] = 0;

				isActive[v] = false;
			}

			visited[v] = false;
		}

		auto BFS = [&](ObjectId v)
		{
			// Mark the current node as visited.
			visited[v] = true;

			for each (auto vid in mReverseEdges[v])
			{
				if (!visited[vid] && !isActive[vid]) {
					activeId.push(vid);
					isActive[vid] = true;
				}
			}

			int l = maxLayer;
			for each (auto vid in mEdges[v])
			{
				l = std::min(mLayers[vid], l);
			}

			mLayers[v] = l - 1;
		};

		while (!activeId.empty())
		{
			ObjectId v = activeId.front();

			bool isReady = true;
			for each (auto w in mEdges[v])
			{
				if (!visited[w])
				{
					isReady = false;
					break;
				}
			}

			if (isReady)
			{
				BFS(v);

				activeId.pop();

				isActive[v] = false;
			}
			else
			{
				activeId.pop();
				
				//Push v into the queue end
				activeId.push(v);
			}

		}

		int minLayler = maxLayer;
		for each (auto v in mVertices)
		{
			minLayler = std::min(minLayler, mLayers[v]);
		}

		for each (auto v in mVertices)
		{
			mLayers[v] -= minLayler;
		}

		mLayerNum = maxLayer - minLayler;
	}

	void AutoLayoutDAG::addDummyVertices()
	{
		ObjectId radixId = Object::baseId();
		for each (auto v in mVertices)
		{
			radixId = std::max(v, radixId);
		}

		radixId++;

		std::map<ObjectId, std::unordered_set<ObjectId>> mNewEdges(mEdges);

		for each (auto it in mEdges)
		{
			ObjectId v = it.first;
			for each (auto w in it.second)
			{
				int edgeLength = mLayers[w] - mLayers[v];

				if (edgeLength > 1)
				{
					mNewEdges[v].erase(w);
					mReverseEdges[w].erase(v);

					ObjectId v0, w0;
					//Insert new edges
					for (int i = 0; i < edgeLength; i++)
					{
						v0 = i == 0 ? v : w0;
						w0 = i == edgeLength - 1 ? w : radixId;

						mVertices.insert(w0);

						mLayers[w0] = mLayers[v] + i + 1;

						// Add an edge (v, w).
						mNewEdges[v0].insert(w0);

						//Add a reverse edge (w, v)
						mReverseEdges[w0].insert(v0);

						radixId++;
					}
				}
			}
		}

		mEdges = mNewEdges;
	}

	struct WNode
	{
		ObjectId id;
		float weight;
	};

	void AutoLayoutDAG::minimizeEdgeCrossings()
	{
		std::vector<std::vector<WNode>> weightedNodes(mLayerNum);

		auto compare_vertex = [=](WNode n0, WNode n1) -> bool
		{
			return n0.weight < n1.weight;
		};

		//Initialize weighted nodes and the x-coordinates for all vertices
		for each (auto  v in mVertices)
		{
			WNode ln = { v, 1.0f };
			
			int layer = mLayers[v];

			mXCoordinate[v] = weightedNodes[layer].size();
			weightedNodes[layer].push_back(ln);
		}

		for (size_t t = 0; t < mIterNum; t++)
		{
			//Reorder each layer from layer i to i + 1
			for (size_t l = 1; l < weightedNodes.size(); l++)
			{
				for (size_t i = 0; i < weightedNodes[l].size(); i++)
				{
					float sum = 0.0f;
					ObjectId v = weightedNodes[l][i].id;
					for each (auto w in mReverseEdges[v])
					{
						sum += mXCoordinate[w];
					}

					weightedNodes[l][i] = { v, sum / mReverseEdges[v].size() };
				}

				std::sort(weightedNodes[l].begin(), weightedNodes[l].end(), compare_vertex);

				for (size_t i = 0; i < weightedNodes[l].size(); i++)
				{
					ObjectId v = weightedNodes[l][i].id;

					mXCoordinate[v] = i;
				}
			}

			//Reorder each layer from layer i + 1 to i
			for (size_t l = mLayerNum - 2; l < mLayerNum; l--)
			{
				for (size_t i = 0; i < weightedNodes[l].size(); i++)
				{
					float sum = 0.0f;
					ObjectId v = weightedNodes[l][i].id;
					for each (auto w in mEdges[v])
					{
						sum += mXCoordinate[w];
					}

					weightedNodes[l][i] = { v, sum / mEdges[v].size() };
				}

				std::sort(weightedNodes[l].begin(), weightedNodes[l].end(), compare_vertex);

				for (size_t i = 0; i < weightedNodes[l].size(); i++)
				{
					ObjectId v = weightedNodes[l][i].id;

					mXCoordinate[v] = i;
				}
			}
		}
		
		mNodeLayers.resize(mLayerNum);

		//Construct layers
		for (size_t l = 0; l < weightedNodes.size(); l++)
		{
			for each (auto n in weightedNodes[l])
			{
				mNodeLayers[l].push_back(n.id);
			}
		}

		for (size_t l = 0; l < weightedNodes.size(); l++)
		{
			weightedNodes[l].clear();
		}

		weightedNodes.clear();
	}
}