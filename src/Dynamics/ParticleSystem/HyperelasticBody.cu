#include "HyperelasticBody.h"
#include "Topology/UnstructuredPointSet.h"
#include "Utility.h"
#include "Topology/NeighborQuery.h"
#include "ParticleIntegrator.h"
#include "HyperelasticityModule_test.h"

#include "Smesh_IO/smesh.h"

#include <thrust/sort.h>

namespace dyno
{
	IMPLEMENT_CLASS_1(HyperelasticBody, TDataType)

	template<typename TDataType>
	HyperelasticBody<TDataType>::HyperelasticBody(std::string name)
		: ParticleSystem<TDataType>(name)
	{
		m_pSet = std::make_shared<TetrahedronSet<TDataType>>();
		this->setTopologyModule(m_pSet);

		this->varHorizon()->setValue(0.0085);
		//		this->attachField(&m_horizon, "horizon", "horizon");

		m_integrator = this->template setNumericalIntegrator<ParticleIntegrator<TDataType>>("integrator");
		this->currentPosition()->connect(m_integrator->inPosition());
		this->currentVelocity()->connect(m_integrator->inVelocity());
		this->currentAttribute()->connect(m_integrator->inAttribute());
		this->currentForce()->connect(m_integrator->inForceDensity());

		this->getAnimationPipeline()->push_back(m_integrator);


		m_hyper = this->template addConstraintModule<HyperelasticityModule_test<TDataType>>("elasticity");
		this->varHorizon()->connect(m_hyper->inHorizon());
		this->currentPosition()->connect(m_hyper->inPosition());
		this->currentVelocity()->connect(m_hyper->inVelocity());
		this->currentRestShape()->connect(m_hyper->inRestShape());
		this->currentVolume()->connect(m_hyper->inVolume());
		this->currentAttribute()->connect(m_hyper->inAttribute());
		this->currentVertexRotation()->connect(m_hyper->inRotation());
		this->getAnimationPipeline()->push_back(m_hyper);

		//Create a node for surface mesh rendering
// 		m_mesh_node = std::make_shared<TetSystem<TDataType>>("Mesh");
// 		this->addChild(m_mesh_node);

		//Set the topology mapping from PointSet to TriangleSet
	}

	template<typename TDataType>
	HyperelasticBody<TDataType>::~HyperelasticBody()
	{
		
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::translate(Coord t)
	{
		m_pSet->translate(t);

		return true;
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::scale(Real s)
	{
		m_pSet->scale(s);

		//this->getMeshNode()->scale(s);
		this->varHorizon()->setValue(s*this->varHorizon()->getValue());

		return true;
	}


	template<typename TDataType>
	bool HyperelasticBody<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::advance(Real dt)
	{
		m_integrator->begin();

		m_integrator->integrate();

		m_hyper->update();
		

		m_integrator->end();
	}

	//夏提完成，根据UnstructuredPointSet中m_coords和m_pointNeighbors来更新TetrahedronSet
	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateTopology()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (tetSet == nullptr) return;

		if (!this->currentPosition()->isEmpty())
		{
			int num = this->currentPosition()->getElementCount();
			auto& pts = m_pSet->getPoints();
			if (num != pts.size())
			{
				pts.resize(num);
			}

			Function1Pt::copy(pts, this->currentPosition()->getValue());
		}
	}

	template<typename Matrix>
	__global__ void InitRotation(
		GArray<Matrix> rots)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= rots.size()) return;

		rots[pId] = Matrix::identityMatrix();
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::resetStatus()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (tetSet == nullptr) return false;

		CArray<Attribute> host_attribute;
		host_attribute.resize(tetSet->getPoints().size());
		for (int i = 0; i < tetSet->getPoints().size(); i++)
		{
			host_attribute[i] = Attribute();
		}

		int vNum = tetSet->getPoints().size();

		this->currentRestPosition()->setElementCount(vNum);
		Function1Pt::copy(this->currentRestPosition()->getValue(), tetSet->getPoints());

		this->currentAttribute()->setElementCount(vNum);
		Function1Pt::copy(this->currentAttribute()->getValue(), host_attribute);

		this->currentVertexRotation()->setElementCount(vNum);
		cuExecute(vNum,
			InitRotation,
			this->currentVertexRotation()->getValue());

		host_attribute.release();

		this->updateVolume();
		this->updateRestShape();

		/*CArray<Coord> host_position;
		CArray<TopologyModule::Tetrahedron> host_tets;
		host_position.resize(tetSet->getPoints().size());
		host_tets.resize(tetSet->getTetrahedrons().size());

		std::vector<Coord> stl_position;
		std::vector<TopologyModule::Tetrahedron> stl_tets;

		Function1Pt::copy(host_position, tetSet->getPoints());
		Function1Pt::copy(host_tets, tetSet->getTetrahedrons());

		for (int i = 0; i < host_position.size(); i++)
		{
			stl_position.push_back(host_position[i]);
		}
		for (int i = 0; i < host_tets.size(); i++)
		{
			stl_tets.push_back(host_tets[i]);
		}

		updateRestShape(stl_position, stl_tets);*/




		return ParticleSystem<TDataType>::resetStatus();
	}

	template<typename TDataType>
	std::shared_ptr<ElasticityModule<TDataType>> HyperelasticBody<TDataType>::getElasticitySolver()
	{
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");
		return module;
	}


	template<typename TDataType>
	void HyperelasticBody<TDataType>::setElasticitySolver(std::shared_ptr<ElasticityModule<TDataType>> solver)
	{
		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");
		auto module = this->template getModule<ElasticityModule<TDataType>>("elasticity");

		this->currentPosition()->connect(solver->inPosition());
		this->currentVelocity()->connect(solver->inVelocity());
		nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
		this->varHorizon()->connect(solver->inHorizon());

		this->deleteModule(module);
		
		solver->setName("elasticity");
		this->addConstraintModule(solver);
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadCentroidsFromFile(std::string filename)
	{
		Smesh meshLoader;
		meshLoader.loadNodeFile(filename + ".node");
		meshLoader.loadTetFile(filename + ".ele");

		this->getMeshNode()->derivedTopology()->setPoints(meshLoader.m_points);
		this->getMeshNode()->derivedTopology()->setTetrahedrons(meshLoader.m_tets);

		std::vector<Vector3f> centroids;
		centroids.resize(meshLoader.m_tets.size());

		for (int i= 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];
			centroids[i] = (meshLoader.m_points[tet[0]] + meshLoader.m_points[tet[1]] + meshLoader.m_points[tet[2]] + meshLoader.m_points[tet[3]]) / 4;
			//printf("centroids %d: %f %f %f \n", i, centroids[i][0], centroids[i][1], centroids[i][2]);
		}

		float global_min = 1000000;
		for (int i = 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];

			Vector3f v0 = meshLoader.m_points[tet[0]];
			Vector3f v1 = meshLoader.m_points[tet[1]];
			Vector3f v2 = meshLoader.m_points[tet[2]];
			Vector3f v3 = meshLoader.m_points[tet[3]];

			Vector3f min_v = v0.minimum(v1).minimum(v2.minimum(v3));
			Vector3f max_v = v0.maximum(v1).maximum(v2.maximum(v3));

			Vector3f bounding = max_v - min_v;

			float max_edge = maximum(maximum(bounding[0], bounding[1]), bounding[2]);

			global_min = max_edge < global_min ? max_edge : global_min;
		}

		DynamicArray<int> vertexToTetList;
		vertexToTetList.resize(meshLoader.m_points.size());

		for (int i = 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];

			for (int j = 0; j < 4; j++)
			{
				vertexToTetList[tet[j]].insert(i);
			}
		}


		//construct neighboring tets for each tet
		DynamicArray<int> tetToTetList;
		tetToTetList.resize(meshLoader.m_tets.size());

		auto findNeighborTetId = [&](int tetId, int vId0, int vId1, int vId2) -> int {
			std::map<int, int> tet_num;
			for each (auto tetId in vertexToTetList[vId0])
			{
				tet_num[tetId] = 0;
			}

			for each (auto tetId in vertexToTetList[vId1])
			{
				tet_num[tetId] = 0;
			}

			for each (auto tetId in vertexToTetList[vId2])
			{
				tet_num[tetId] = 0;
			}

			//statistics
			for each (auto tetId in vertexToTetList[vId0])
			{
				tet_num[tetId] += 1;
			}

			for each (auto tetId in vertexToTetList[vId1])
			{
				tet_num[tetId] += 1;
			}

			for each (auto tetId in vertexToTetList[vId2])
			{
				tet_num[tetId] += 1;
			}

			for each (auto var in tet_num)
			{
				if (var.first != tetId && var.second == 3)
				{
					return var.first;
				}
			}

			tet_num.clear();

			return -1;
		};

		for (int i = 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];

			int tId0 = findNeighborTetId(i, tet[0], tet[1], tet[2]);
			int tId1 = findNeighborTetId(i, tet[1], tet[2], tet[3]);
			int tId2 = findNeighborTetId(i, tet[2], tet[3], tet[0]);
			int tId3 = findNeighborTetId(i, tet[3], tet[0], tet[1]);

			if (tId0 >= 0)	tetToTetList[i].insert(tId0);
			if (tId1 >= 0)	tetToTetList[i].insert(tId1);
			if (tId2 >= 0)	tetToTetList[i].insert(tId2);
			if (tId3 >= 0)	tetToTetList[i].insert(tId3);
		}

		
		typedef TPair<TDataType> NPair;

		std::vector<NPair> elements;
		std::vector<int> index;

		int offset = 0;
		for(int i = 0; i < tetToTetList.size(); i++)
		{
			auto& var = tetToTetList[i];
	
			elements.push_back(NPair(i, centroids[i]));
			printf("%d center: %f %f %f \n", i, centroids[i][0], centroids[i][1], centroids[i][2]);
			for each (auto j in var)
			{
				elements.push_back(NPair(j, centroids[j]));

				printf("pair: %d %d; %f %f %f \n", i, j, centroids[j][0], centroids[j][1], centroids[j][2]);
			}

			index.push_back(offset);
			offset += (var.size() + 1);
		}


 		this->currentRestShape()->setElementCount(index.size());
 		this->currentRestShape()->getReference()->copyFrom(0, elements, index);

		this->varHorizon()->setValue(2.0*global_min);

		m_pSet->setPoints(centroids);
	}


	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadVertexFromFile(std::string filename)
	{
		Smesh meshLoader;
		meshLoader.loadNodeFile(filename + ".node");
		meshLoader.loadTriangleFile(filename + ".face");
		meshLoader.loadTetFile(filename + ".ele");

// 		this->getMeshNode()->derivedTopology()->setPoints(meshLoader.m_points);
// 		this->getMeshNode()->derivedTopology()->setTetrahedrons(meshLoader.m_tets);

		std::vector<Vector3f> centroids;
		centroids.resize(meshLoader.m_tets.size());

		for (int i = 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];
			centroids[i] = (meshLoader.m_points[tet[0]] + meshLoader.m_points[tet[1]] + meshLoader.m_points[tet[2]] + meshLoader.m_points[tet[3]]) / 4;
			//printf("centroids %d: %f %f %f \n", i, centroids[i][0], centroids[i][1], centroids[i][2]);
		}

		float global_min = 1000000;
		for (int i = 0; i < meshLoader.m_tets.size(); i++)
		{
			auto tet = meshLoader.m_tets[i];

			Vector3f v0 = meshLoader.m_points[tet[0]];
			Vector3f v1 = meshLoader.m_points[tet[1]];
			Vector3f v2 = meshLoader.m_points[tet[2]];
			Vector3f v3 = meshLoader.m_points[tet[3]];

			Vector3f min_v = v0.minimum(v1).minimum(v2.minimum(v3));
			Vector3f max_v = v0.maximum(v1).maximum(v2.maximum(v3));

			Vector3f bounding = max_v - min_v;

			float max_edge = maximum(maximum(bounding[0], bounding[1]), bounding[2]);

			global_min = max_edge < global_min ? max_edge : global_min;
		}

		this->varHorizon()->setValue(2.0*global_min);

		m_pSet->setPoints(meshLoader.m_points);
		//m_pSet->setTriangles(meshLoader.m_triangles);
		m_pSet->setTetrahedrons(meshLoader.m_tets);
	}


	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateRestShape(std::vector<Coord>& points, std::vector<TopologyModule::Tetrahedron>& tets)
	{
		DynamicArray<int> vertToVertList;
		vertToVertList.resize(points.size());

		for (int i = 0; i < tets.size(); i++)
		{
			auto tet = tets[i];

			for (int j = 0; j < 4; j++)
			{
				int v0 = tet[j];
				int v1 = tet[(j + 1) % 4];

				vertToVertList[v0].insert(v1);
				vertToVertList[v1].insert(v0);
			}

			int v0 = tet[0];
			int v1 = tet[2];

			vertToVertList[v0].insert(v1);
			vertToVertList[v1].insert(v0);

			v0 = tet[1];
			v1 = tet[3];

			vertToVertList[v0].insert(v1);
			vertToVertList[v1].insert(v0);
		}

		std::vector<NPair> elements;
		std::vector<int> index;

		int offset = 0;
		for (int i = 0; i < vertToVertList.size(); i++)
		{
			auto& var = vertToVertList[i];

			elements.push_back(NPair(i, points[i]));
			for each (auto j in var)
			{
				elements.push_back(NPair(j, points[j]));

				//printf("pair: %d %d \n", i, j);
			}

			index.push_back(offset);
			offset += (var.size() + 1);
		}


		this->currentRestShape()->setElementCount(index.size());
		this->currentRestShape()->getReference()->copyFrom(0, elements, index);

		elements.clear();
		index.clear();
	}

	__global__ void SetSize(
		GArray<int> index,
		ArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;

		index[pId] = lists[pId].size() + 1;
	}

	template<typename Coord, typename NPair>
	__global__ void SetRestShape(
		GArray<NPair> elements,
		GArray<int> shifts,
		GArray<Coord> restPos,
		ArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;

		int shift = shifts[pId];
		int index = 1;

		elements[shift] = NPair(pId, restPos[pId]);
		
		List<int> list = lists[pId];
		for (auto it = list.begin(); it != list.end(); it++)
		{
			int j = *it;
			elements[shift + index] = NPair(j, restPos[j]);

			index++;
		}
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateRestShape()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (tetSet == nullptr) return;

		auto neighbors = tetSet->getPointNeighbors();

		auto& restPos = this->currentRestPosition()->getValue();

		this->currentRestShape()->setElementCount(restPos.size());

		auto& index = this->currentRestShape()->getReference()->getIndex();
		auto& elements = this->currentRestShape()->getReference()->getElements();

		cuExecute(neighbors->size(),
			SetSize,
			index,
			*neighbors);

		int total_num = thrust::reduce(thrust::device, index.begin(), index.begin() + index.size());
		thrust::exclusive_scan(thrust::device, index.begin(), index.begin() + index.size(), index.begin());

		elements.resize(total_num);

		cuExecute(neighbors->size(),
			SetRestShape,
			elements,
			index,
			restPos,
			*neighbors);
	}

	template<typename Real, typename Coord, typename Tetrahedron>
	__global__ void HB_CalculateVolume(
		GArray<Real> volume,
		GArray<Coord> restPos,
		GArray<Tetrahedron> tets,
		NeighborList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;

		Real vol_i = Real(0);

		int ne = lists.getNeighborSize(pId);
		for (int i = 0; i < ne; i++)
		{
			int tetId = lists.getElement(pId, i);
			Tetrahedron tetIndex = tets[tetId];

			TTet3D<Real> tet(restPos[tetIndex[0]], restPos[tetIndex[1]], restPos[tetIndex[2]], restPos[tetIndex[3]]);

			vol_i += tet.volume();
		}

		volume[pId] = maximum(vol_i, Real(0.0001));
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateVolume()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (tetSet == nullptr) return;

		auto& ver2Tet = tetSet->getVer2Tet();

		auto& restPos = this->currentRestPosition()->getValue();

		this->currentVolume()->setElementCount(restPos.size());

		cuExecute(restPos.size(),
			HB_CalculateVolume,
			this->currentVolume()->getValue(),
			restPos,
			tetSet->getTetrahedrons(),
			ver2Tet);

		auto& volume = this->currentVolume()->getValue();

		Reduction<Real> reduce;
		Real max_vol = reduce.maximum(volume.begin(), volume.size());
		Real min_vol = reduce.minimum(volume.begin(), volume.size());

		printf("max vol: %f; min vol: %f \n", max_vol, min_vol);
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadStandardTet()
	{
		std::vector<Vector3f> centroids;
		centroids.resize(4);

		centroids[0] = Vector3f(0.101f, 0.1, 0.1);
		centroids[1] = Vector3f(0.105f, 0.1, 0.1);
		centroids[2] = Vector3f(0.1f, 0.095f, 0.1);
		centroids[3] = Vector3f(0.1f, 0.1, 0.105f);

		std::vector<NPair> elements;
		std::vector<int> index;

		int offset = 0;
		for (int i = 0; i < centroids.size(); i++)
		{
			elements.push_back(NPair(i, centroids[i]));
			for(int j = i; j < i + 3; j++)
			{
				int ne = (j + 1) % 4;
				elements.push_back(NPair(ne, centroids[ne]));
			}

			index.push_back(offset);
			offset += 4;
		}

		this->currentRestShape()->setElementCount(index.size());
		this->currentRestShape()->getReference()->copyFrom(0, elements, index);

//		this->varHorizon()->setValue(0.006);

		std::vector<TopologyModule::Tetrahedron> tets;
		tets.push_back(TopologyModule::Tetrahedron(0, 1, 2, 3));

		m_pSet->setPoints(centroids);
		m_pSet->setTetrahedrons(tets);
	}


	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadStandardSimplex()
	{
		std::vector<Vector3f> centroids;
		centroids.resize(5);

		centroids[0] = Vector3f(0.101f, 0.1, 0.1);
		centroids[1] = Vector3f(0.105f, 0.1, 0.1);
		centroids[2] = Vector3f(0.1f, 0.095f, 0.1);
		centroids[3] = Vector3f(0.1f, 0.1, 0.105f);
		centroids[4] = Vector3f(0.1f, 0.1, 0.11f);

		std::vector<NPair> elements;
		std::vector<int> index;

		int offset = 0;
		for (int i = 0; i < centroids.size() - 1; i++)
		{
			elements.push_back(NPair(i, centroids[i]));
			for (int j = i; j < i + 3; j++)
			{
				int ne = (j + 1) % 4;
				elements.push_back(NPair(ne, centroids[ne]));
			}

			index.push_back(offset);
			offset += 4;
		}

		elements.push_back(NPair(4, centroids[4]));
		offset++;

		index.push_back(offset);
		
		elements.push_back(NPair(4, centroids[4]));
//		elements.push_back(NPair(3, centroids[3]));

		this->currentRestShape()->setElementCount(index.size());
		this->currentRestShape()->getReference()->copyFrom(0, elements, index);

		this->varHorizon()->setValue(0.006);

		m_pSet->setPoints(centroids);

		/*std::vector<Vector3f> centroids;
		centroids.resize(4);

		centroids[0] = Vector3f(0.500000, 0.750000, 0.500000);
		centroids[1] = Vector3f(0.250000, 0.750000, 0.250000);
		centroids[2] = Vector3f(0.250000, 0.500000, 0.750000);
		centroids[3] = Vector3f(0.750000, 0.500000, 0.250000);

		std::vector<NPair> elements;
		std::vector<int> index;

		elements.push_back(NPair(0, centroids[0]));
		elements.push_back(NPair(1, centroids[1]));
		elements.push_back(NPair(2, centroids[2]));
		elements.push_back(NPair(3, centroids[3]));

		index.push_back(0);
		index.push_back(4);
		index.push_back(4);
		index.push_back(4);


		this->currentRestShape()->setElementCount(index.size());
		this->currentRestShape()->getReference()->copyFrom(0, elements, index);

		this->varHorizon()->setValue(0.5);

		m_pSet->setPoints(centroids);*/
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadParticles(Coord lo, Coord hi, Real distance)
	{
		std::vector<Coord> vertList;

		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real y = lo[1]; y <= hi[1]; y += distance)
			{
				for (Real z = lo[2]; z <= hi[2]; z += distance)
				{
					Coord p = Coord(x, y, z);
					vertList.push_back(Coord(x, y, z));
				}
			}
		}
		m_pSet->setPoints(vertList);

		std::cout << "particle number: " << vertList.size() << std::endl;

		vertList.clear();
	}
}