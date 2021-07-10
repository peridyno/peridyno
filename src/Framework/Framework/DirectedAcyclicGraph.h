#include <map>
#include <list>
#include <stack>
#include <set>
#include <vector>

#include "Object.h"

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

	private:
		// Functions used by topologicalSort
		void topologicalSortUtil(ObjectId v, std::map<ObjectId, bool>& visited, std::stack<ObjectId>& stack);
		void topologicalSortUtil(ObjectId v, std::map<ObjectId, bool>& visited);

	private:
		std::vector<ObjectId> mOrderVertices;
		std::set<ObjectId> mVertices;
		std::map<ObjectId, std::list<ObjectId>> mEdges;
	};
}
