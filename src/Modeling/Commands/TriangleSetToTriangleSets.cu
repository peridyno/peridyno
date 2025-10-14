#include "TriangleSetToTriangleSets.h"
#include "Topology/TriangleSets.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"
#include <unordered_set>
#include <unordered_map>

namespace dyno
{
	template<typename TDataType>
	TriangleSetToTriangleSets<TDataType>::TriangleSetToTriangleSets()
		: ParametricModel<TDataType>()
	{
		this->stateTriangleSets()->setDataPtr(std::make_shared<TriangleSets<TDataType>>());

		auto glModule = std::make_shared<GLSurfaceVisualModule>();
		glModule->setColor(Color(0.8f, 0.52f, 0.25f));
		glModule->setVisible(true);
		this->stateTriangleSets()->connect(glModule->inTriangleSet());
		this->graphicsPipeline()->pushModule(glModule);

		auto glModule2 = std::make_shared<GLPointVisualModule>();
		glModule2->setColor(Color(1.0f, 0.1f, 0.1f));
		glModule2->setVisible(false);
		glModule2->varPointSize()->setValue(0.01);
		this->stateTriangleSets()->connect(glModule2->inPointSet());
		this->graphicsPipeline()->pushModule(glModule2);

		auto glModule3 = std::make_shared<GLWireframeVisualModule>();
		glModule3->setColor(Color(0.0f, 0.0f, 0.0f));
		glModule3->setVisible(false);
		this->stateTriangleSets()->connect(glModule3->inEdgeSet());
		this->graphicsPipeline()->pushModule(glModule3);

		this->stateTriangleSets()->promoteOuput();

	}

	template<typename TDataType>
	void TriangleSetToTriangleSets<TDataType>::resetStates()
	{
		if (this->inTriangleSet()->isEmpty())
			return;

		std::vector<std::vector<int>> ConnectivityResult = groupTrianglesByConnectivity(this->inTriangleSet()->getDataPtr());

		auto triSet = this->inTriangleSet()->getDataPtr();
		auto triSets = this->stateTriangleSets()->getDataPtr();

		triSets->setPoints(triSet->getPoints());
		triSets->setTriangles(triSet->triangleIndices());
		triSets->setShapeSize(ConnectivityResult.size());

		CArray<uint> c_ShapeIds(triSet->triangleIndices().size());

		for (size_t i = 0; i < ConnectivityResult.size(); i++)
		{
			for (auto triId : ConnectivityResult[i])
			{
				c_ShapeIds[triId] = i;
			}
		}
		triSets->shapeIds().assign(c_ShapeIds);
	}

	template<typename TDataType>
	std::vector<std::vector<int>> TriangleSetToTriangleSets<TDataType>::groupTrianglesByConnectivity(std::shared_ptr<TriangleSet<TDataType>> triSet)
	{
		std::unordered_map<int, std::vector<int>> vertex_to_triangles;

		CArray<TopologyModule::Triangle> triangles;
		triangles.assign(triSet->triangleIndices());

		int n = triangles.size();

		CArrayList<int>ver2TriList;
		ver2TriList.assign(triSet->vertex2Triangle());

		std::vector<std::vector<int>> adjacency(n);		// TirId-{Neighbors: TriId}

		for (int i = 0; i < n; i++) {
			std::unordered_set<int> neighbors;
			for (int j = 0; j < 3; j++)
			{
				int v = triangles[i][j];
				dyno::List<int> TriList = ver2TriList[v];

				for (auto t_idx : TriList)
				{
					if (t_idx != i) {
						neighbors.insert(t_idx);
					}
				}
			}
			adjacency[i] = std::vector<int>(neighbors.begin(), neighbors.end());
		}

		std::vector<bool> visited(n, false);
		std::vector<std::vector<int>> groups;

		for (int i = 0; i < n; i++) {
			if (!visited[i]) // == false : Find a new group
			{
				std::vector<int> group;
				std::queue<int> q;
				q.push(i);	
				visited[i] = true;

				// BFS :  Update current group 
				while (!q.empty()) {
					int curr = q.front(); q.pop();
					group.push_back(curr);

					for (int nei : adjacency[curr])
					{
						if (!visited[nei]) {
							visited[nei] = true;
							q.push(nei);
						}
					}
				}
				groups.push_back(group);
			}
		}

		return groups;
	};

	DEFINE_CLASS(TriangleSetToTriangleSets);
}