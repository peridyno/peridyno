#include "DirectedAcyclicGraph.h"

namespace dyno {

	DirectedAcyclicGraph::~DirectedAcyclicGraph()
	{
		for each (auto edge in mEdges)
		{
			edge.second.clear();
		}
		mEdges.clear();

		for each (auto edge in mReverseEdges)
		{
			edge.second.clear();
		}
		mReverseEdges.clear();

		mVertices.clear();
		mOrderVertices.clear();
	}

	void DirectedAcyclicGraph::addEdge(ObjectId v, ObjectId w)
	{
		mVertices.insert(v);
		mVertices.insert(w);

		// Add an edge (v, w).
		mEdges[v].insert(w); 

		//Add a reverse edge (w, v)
		mReverseEdges[w].insert(v);
	}

	std::vector<ObjectId>& DirectedAcyclicGraph::topologicalSort(ObjectId v)
	{
		mOrderVertices.clear();

		int num = sizeOfVertex();
		// Mark all the vertices as not visited
		std::map<ObjectId, bool> visited;
		for each(auto id in mVertices)
		{
			visited[id] = false;
		}

		topologicalSortUtil(v, visited);

		visited.clear();

		return mOrderVertices;
	}

	std::vector<ObjectId>& DirectedAcyclicGraph::topologicalSort()
	{
		mOrderVertices.clear();

		std::stack<ObjectId> stack;

		int num = sizeOfVertex();
		// Mark all the vertices as not visited
		std::map<ObjectId, bool> visited;
		for each(auto id in mVertices)
		{
			visited[id] = false;
		}

		for each(auto id in mVertices)
			if (visited[id] == false)
				topologicalSortUtil(id, visited, stack);

		// Output contents of stack
		while (stack.empty() == false) {
			mOrderVertices.push_back(stack.top());
			stack.pop();
		}

		visited.clear();

		return mOrderVertices;
	}

	size_t DirectedAcyclicGraph::sizeOfVertex() const
	{
		return mVertices.size();
	}

	std::set<dyno::ObjectId>& DirectedAcyclicGraph::vertices()
	{
		return mVertices;
	}

	std::map<dyno::ObjectId, std::unordered_set<dyno::ObjectId>>& DirectedAcyclicGraph::edges()
	{
		return mEdges;
	}

	std::map<dyno::ObjectId, std::unordered_set<dyno::ObjectId>>& DirectedAcyclicGraph::reverseEdges()
	{
		return mReverseEdges;
	}

	void DirectedAcyclicGraph::topologicalSortUtil(ObjectId v, std::map<ObjectId, bool>& visited, std::stack<ObjectId>& stack)
	{
		// Mark the current node as visited.
		visited[v] = true;

		std::list<ObjectId>::iterator i;
		
		std::stack<ObjectId> reverseId;
		for each (auto id in mEdges[v])
			reverseId.push(id);

		while ((reverseId.empty() == false)) {
			ObjectId id = reverseId.top();
			if (!visited[id])
				topologicalSortUtil(id, visited, stack);

			reverseId.pop();
		}

		// Push current vertex to stack
		stack.push(v);
	}

	void DirectedAcyclicGraph::topologicalSortUtil(ObjectId v, std::map<ObjectId, bool>& visited)
	{
		// Mark the current node as visited and
		visited[v] = true;
		mOrderVertices.push_back(v);

		// Recur for all the vertices adjacent
		std::list<ObjectId>::iterator i;
		for each(auto id in mEdges[v])
			if (!visited[id])
				topologicalSortUtil(id, visited);
	}

}