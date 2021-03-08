#include "HyperelastoplasticityBody.h"
#include "ParticleSystem/PositionBasedFluidModel.h"

#include "Topology/TriangleSet.h"
#include "Topology/UnstructuredPointSet.h"

#include "Utility.h"
#include "ParticleSystem/Peridynamics.h"
#include "Mapping/PointSetToPointSet.h"
#include "Topology/NeighborQuery.h"
#include "Topology/NeighborTetQuery.h"
#include "ParticleSystem/ParticleIntegrator.h"
#include "ParticleSystem/ElastoplasticityModule.h"
#include "ParticleSystem/HyperelasticityModule_test.h"
#include "ParticleSystem/TetCollision.h"

#include "ParticleSystem/DensityPBD.h"
#include "ParticleSystem/ImplicitViscosity.h"

#include "HyperelasticFractureModule.h"


#include "STL/MultiMap.h"
#include "STL/Set.h"
#include "STL/Pair.h"

#include <thrust/sort.h>

//#define DEBUG_INFO

namespace dyno
{
	IMPLEMENT_CLASS_1(HyperelastoplasticityBody, TDataType)

	template<typename TDataType>
	HyperelastoplasticityBody<TDataType>::HyperelastoplasticityBody(std::string name)
		: HyperelasticBody<TDataType>(name)
	{
		m_plasticity = this->template addConstraintModule<HyperelastoplasticityModule<TDataType>>("elastoplasticity");
		this->currentPosition()->connect(m_plasticity->inPosition());
		this->currentVelocity()->connect(m_plasticity->inVelocity());
		this->currentRestShape()->connect(m_plasticity->inRestShape());
		this->currentAttribute()->connect(m_plasticity->inAttribute());
		this->currentVolume()->connect(m_plasticity->inVolume());
		this->currentPrincipleYielding()->connect(m_plasticity->inPrincipleYielding());
		this->currentVertexRotation()->connect(m_plasticity->inRotation());

		m_fracture = this->template addConstraintModule<HyperelasticFractureModule<TDataType>>("fracture");
		this->currentPosition()->connect(m_fracture->inPosition());
		this->currentRestPosition()->connect(m_fracture->inRestPosition());
		this->currentFractureTag()->connect(m_fracture->inFractureTag());

		m_hyper_new = this->template addConstraintModule<HyperelasticityModule_test<TDataType>>("hyper_new");
		this->currentPosition()->connect(m_hyper_new->inPosition());
		this->currentVelocity()->connect(m_hyper_new->inVelocity());
		this->currentRestShape()->connect(m_hyper_new->inRestShape());
		this->currentAttribute()->connect(m_hyper_new->inAttribute());
		this->currentVolume()->connect(m_hyper_new->inVolume());
		this->currentVertexRotation()->connect(m_hyper_new->inRotation());


		m_linear_elasticity = this->template addConstraintModule<ElasticityModule<TDataType>>("tmp_elasticity");
		this->currentPosition()->connect(m_linear_elasticity->inPosition());
		this->currentVelocity()->connect(m_linear_elasticity->inVelocity());
		this->currentRestShape()->connect(m_linear_elasticity->inRestShape());

		m_pbdModule = this->template addConstraintModule<DensityPBD<TDataType>>("pbd");
		this->varHorizon()->connect(m_pbdModule->varSmoothingLength());
		this->currentPosition()->connect(m_pbdModule->inPosition());
		this->currentVelocity()->connect(m_pbdModule->inVelocity());


		m_visModule = this->template addConstraintModule<ImplicitViscosity<TDataType>>("viscosity");
		m_visModule->setViscosity(Real(1));
		this->varHorizon()->connect(&m_visModule->m_smoothingLength);
		this->currentPosition()->connect(&m_visModule->m_position);
		this->currentVelocity()->connect(&m_visModule->m_velocity);

		
		
		m_nbrTetQuery = this->template addComputeModule<NeighborTetQuery<TDataType>>("tetNeighbor");
		//this->varHorizon()
		this->currentPosition()->connect(m_nbrTetQuery->inPosition());
		m_nbrTetQuery->setTetSet(TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule()));


		m_tetCollision = this->template addCollisionModel<TetCollision<TDataType>>("tetCollision");
		this->currentPosition()->connect(&m_tetCollision->m_position);
	//	this->currentPosition()->connect(&m_tetCollision->m_tet_vertex);
		this->currentVelocity()->connect(&m_tetCollision->m_velocity);
		m_nbrTetQuery->outNeighborhood()->connect(&m_tetCollision->m_neighborhood_tri);
		m_tetCollision->setTetSet(TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule()));
			
		m_surfaceNode = this->template createChild<Node>("Mesh");
		m_surfaceNode->setVisible(false);

		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		m_surfaceNode->setTopologyModule(triSet);

// 		std::shared_ptr<PointSetToPointSet<TDataType>> surfaceMapping = std::make_shared<PointSetToPointSet<TDataType>>(this->m_pSet, triSet);
// 		this->addTopologyMapping(surfaceMapping);
	}

	template<typename TDataType>
	HyperelastoplasticityBody<TDataType>::~HyperelastoplasticityBody()
	{
		
	}


	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::initialize()
	{
		return true;
	}

	__global__ void HB_SetSize(
		GArray<int> index,
		GArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;

		index[pId] = lists[pId].size() + 1;
	}

	template<typename Coord, typename NPair>
	__global__ void HB_SetRestShape(
		GArray<NPair> elements,
		GArray<int> shifts,
		GArray<Coord> restPos,
		GArray<Coord> yieldings,
		GArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;

		Coord yield_i = yieldings[pId];
		SquareMatrix<Real, 3> mat;
		mat(0, 0) = yield_i[0];
		mat(1, 1) = yield_i[1];
		mat(2, 2) = yield_i[2];


		int shift = shifts[pId];
		int index = 1;

		Coord rest_pos_i = restPos[pId];
		elements[shift] = NPair(pId, rest_pos_i);

		List<int> list = lists[pId];
		for (auto it = list.begin(); it != list.end(); it++)
		{
			int j = *it;
			Coord rest_pos_j = restPos[j];
			elements[shift + index] = NPair(j, rest_pos_i + mat * (rest_pos_j - rest_pos_i));

			index++;
		}
	}

	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::updateRestShape()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (tetSet == nullptr) return;

		auto neighbors = tetSet->getPointNeighbors();

		auto& restPos = this->currentRestPosition()->getValue();

		this->currentRestShape()->setElementCount(restPos.size());

		auto& index = this->currentRestShape()->getReference()->getIndex();
		auto& elements = this->currentRestShape()->getReference()->getElements();

		cuExecute(neighbors->size(),
			HB_SetSize,
			index,
			*neighbors);

		int total_num = thrust::reduce(thrust::device, index.begin(), index.begin() + index.size());
		thrust::exclusive_scan(thrust::device, index.begin(), index.begin() + index.size(), index.begin());

		elements.resize(total_num);

		cuExecute(neighbors->size(),
			HB_SetRestShape,
			elements,
			index,
			restPos,
			this->currentPrincipleYielding()->getValue(),
			*neighbors);
	}

	template <typename Coord>
	__global__ void HPB_Rotate(
		GArray<Coord> pos) 
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= pos.size()) return;

		float angle = 10/180.0f;
		Matrix3f rot;
		rot(0, 0) = cos(angle);
		rot(0, 1) = -sin(angle);
		rot(0, 2) = 0;
		rot(1, 0) = sin(angle);
		rot(1, 1) = cos(angle);
		rot(1, 2) = 0;
		rot(2, 0) = 0;
		rot(2, 1) = 0;
		rot(2, 2) = 1;

		pos[pId] = rot * pos[pId];
	}

	int it = 0;
	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::advance(Real dt)
	{
// 		if (it == 0)
// 		{
// 			cuExecute(this->currentPosition()->getElementCount(),
// 				HPB_Rotate,
// 				this->currentPosition()->getValue());
// 
// 			it++;
// 		}

		m_integrator->begin();

		m_integrator->integrate();
		
//		m_plasticity->update();

// 		m_linear_elasticity->update();
 		m_hyper_new->update();

		m_fracture->update();

		if (this->varCollisionEnabled()->getValue() == true)
		{
			GTimer t3;
			t3.start();
			printf("000\n");
			m_nbrTetQuery->compute();
			printf("111\n");



			m_tetCollision->dt = dt;
			m_tetCollision->doCollision();

			t3.stop();

			printf("step 2 time: %f\n", t3.getEclipsedTime());

			printf("222\n");
		}

		//m_hyper->update();

		//m_fracture->updateTopology();

		//m_nbrQuery->compute();
		//m_fracture->solveElasticity();

		/*m_nbrQuery->compute();*/

		//m_pbdModule->constrain();

		//m_fracture->applyPlasticity();

		//m_visModule->constrain();

		m_integrator->end();
	}


	__global__ void HFM_SplitVertex(
		GArray<int> splitNum,
		NeighborList<Pair<int, int>> vertPairs,
		GArray<bool> tagSplit,
		NeighborList<int> ver2Tri,
		NeighborList<int> ver2Tet,
		GArray<TopologyModule::Tri2Tet> tri2Tet,
		GArray<TopologyModule::Tetrahedron> tets,
		GArray<TopologyModule::Triangle> triangles,
		int sharedSkip)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= ver2Tri.size()) return;

		int tId = threadIdx.x;

		extern __shared__ char sBuf[];

		int tet_num = ver2Tet.getNeighborSize(vId);
		int tri_num = ver2Tri.getNeighborSize(vId);

		char* st_addr = sBuf + tId * sharedSkip * (sizeof(Pair<int, int>) + sizeof(int));

		Pair<int, int>* id = (Pair<int, int>*)st_addr;
		MultiMap<int, int> multiMap;
		multiMap.reserve(id, sharedSkip);
		for (int i = 0; i < tet_num; i++)
		{
			int tetId = ver2Tet.getElement(vId, i);
			multiMap.insert(Pair<int, int>(tetId, tetId));

// 			if (vId == 0)
// 			{
// 				printf("before: %d; total size: %d \n", tetId, (int)multiMap.size());
// 			}
		}
		// 
		// 		if (vId == 5)
		// 		{
		// 			for (auto it = multiMap.begin(); it != multiMap.end(); it++)
		// 			{
		// 				printf("after: %d; total size: %u \n", it->second, (int)multiMap.size());
		// 			}
		// 		}

				//merge regions, the computation complexity is O(tri_num^2)
		for (int i = 0; i < tri_num; i++)
		{
			int triId = ver2Tri.getElement(vId, i);

			TopologyModule::Tri2Tet t2t = tri2Tet[triId];
			int tId0 = t2t[0];
			int tId1 = t2t[1];
			if (tagSplit[triId] == false && tId0 != EMPTY && tId1 != EMPTY)
			{
				int tag0 = multiMap[tId0];
				int tag1 = multiMap[tId1];

				int tagE0 = tag0 < tag1 ? tag0 : tag1;
				int tagE1 = tag1 > tag0 ? tag1 : tag0;

				for (auto it = multiMap.begin(); it != multiMap.end(); it++)
				{
					if (it->second == tagE1)
						it->second = tagE0;
				}


// 				if (vId == 0)
// 				{
// 					printf("Test: %d %d %d \n", triId, tId0, tId1);
// 				}
			}
		}

		Set<int> set;
		int* setBuf = (int*)(st_addr + sharedSkip * sizeof(Pair<int, int>));
		set.reserve(setBuf, sharedSkip);

		for (auto it = multiMap.begin(); it != multiMap.end(); it++)
		{
			set.insert(it->second);
// 			if (vId == 0)
// 			{
// 				printf("tetId: %d \n", it->second);
// 			}
		}

		for (auto it = multiMap.begin(); it != multiMap.end(); it++)
		{
			int ind = 0;
			int val = it->second;
			for (auto it_set = set.begin(); it_set != set.end(); it_set++)
			{
				if (*it_set == val)
					break;

				ind++;
			}

			it->second = ind;
		}

		int ne = 0;
		for (auto it = multiMap.begin(); it != multiMap.end(); it++)
		{
			vertPairs.setElement(vId, ne, *it);
			ne++;
		}

		splitNum[vId] = set.size() == 0 ? 1 : set.size();

		//printf("%d %d \n", vId, splitNum[vId]);
	}


	__global__ void HFM_UpdateTopology(
		GArray<TopologyModule::Tetrahedron> tets,
		GArray<TopologyModule::Tetrahedron> tets_old,
		GArray<int> vertNum,
		GArray<int> shift,
		NeighborList<Pair<int, int>> vertPairs)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= vertPairs.size()) return;

		int size = vertPairs.getNeighborSize(vId);

		int shift_v = shift[vId];
//		printf("%d: counter %d; size: %d \n", vId, vertNum[vId], size);
		for (int j = 0; j < size; j++)
		{
			Pair<int, int> pair = vertPairs.getElement(vId, j);

// 			if (vId == 497)
// 			{
// 				printf("Pair %d: %d %d \n", vId, pair.first, pair.second);
// 			}

			int vId_new = shift_v + pair.second;
			TopologyModule::Tetrahedron tet = tets_old[pair.first];
			for (int t = 0; t < 4; t++)
			{
				if (tet[t] == vId)
				{
					tets[pair.first][t] = vId_new;
				}
			}
		}
	}

	template<typename Coord, typename Matrix>
	__global__ void HFM_UpdateFields(
		GArray<Coord> rest_pos_new,
		GArray<Coord> rest_pos_old,
		GArray<Coord> pos_new,
		GArray<Coord> pos_old,
		GArray<Coord> vel_new,
		GArray<Coord> vel_old,
		GArray<Coord> yielding_new,
		GArray<Coord> yielding_old,
		GArray<Attribute> att_new,
		GArray<Attribute> att_old,
		GArray<Matrix> rot_new,
		GArray<Matrix> rot_old,
		GArray<int> vertNum,
		GArray<int> shift)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= pos_old.size()) return;

		int num = vertNum[vId];
		int shift_v = shift[vId];
		Coord pos_v = pos_old[vId];
		Coord rest_pos_v = rest_pos_old[vId];
		Coord vel_v = vel_old[vId];
		Coord yield_v = yielding_old[vId];
		Attribute att_v = att_old[vId];
		Matrix rot_v = rot_old[vId];
		for (int i = 0; i < num; i++)
		{
			pos_new[shift_v + i] = pos_v;
			vel_new[shift_v + i] = vel_v;

			rest_pos_new[shift_v + i] = rest_pos_v;

			att_new[shift_v + i] = att_v;
			yielding_new[shift_v + i] = yield_v;

			rot_new[shift_v + i] = rot_v;
		}
	}

	void print(NeighborList<int>& list)
	{
		CArray<int> index;
		CArray<int> elements;
		index.resize(list.getIndex().size());
		elements.resize(list.getElements().size());

		Function1Pt::copy(index, list.getIndex());
		Function1Pt::copy(elements, list.getElements());

		for (int i = 0; i < index.size(); i++)
		{
			int ne = i == index.size() - 1 ? elements.size() - index[index.size() - 1] : index[i + 1] - index[i];
			for (int j = 0; j < ne; j++)
			{
				printf("%d: %d \n", i, elements[index[i] + j]);
			}
		}

		index.clear();
		elements.clear();
	}

	template<typename NPair>
	void print(NeighborList<NPair>& list)
	{
		CArray<int> index;
		CArray<NPair> elements;
		index.resize(list.getIndex().size());
		elements.resize(list.getElements().size());

		Function1Pt::copy(index, list.getIndex());
		Function1Pt::copy(elements, list.getElements());

		for (int i = 0; i < index.size(); i++)
		{
			int ne = i == index.size() - 1 ? elements.size() - index[index.size() - 1] : index[i + 1] - index[i];
			for (int j = 0; j < ne; j++)
			{
				printf("%d: %d \n", i, elements[index[i] + j].index);
			}
		}

		index.clear();
		elements.clear();
	}

	void print(GArray<TopologyModule::Tetrahedron>& tets)
	{
		CArray<TopologyModule::Tetrahedron> h_tets;
		h_tets.resize(tets.size());
		Function1Pt::copy(h_tets, tets);

		for (size_t i = 0; i < tets.size(); i++)
		{
			printf("Tet %d: %d %d %d %d \n", i, h_tets[i][0], h_tets[i][1], h_tets[i][2], h_tets[i][3]);
		}

		h_tets.clear();
	}

// 	void print(GArray<bool>& bArray)
// 	{
// 		CArray<bool> h_bool;
// 		h_bool.resize(bArray.size());
// 		Function1Pt::copy(h_bool, bArray);
// 
// 		for (size_t i = 0; i < h_bool.size(); i++)
// 		{
// 			if (h_bool[i])
// 			{
// 				printf("Array %d: true \n", i);
// 			}
// 			else
// 			{
// 				printf("Array %d: false \n", i);
// 			}
// 		}
// 
// 		h_bool.release();
// 	}

	void print(GArray<int>& intArray)
	{
		CArray<int> h_intArray;
		h_intArray.resize(intArray.size());
		Function1Pt::copy(h_intArray, intArray);

		for (size_t i = 0; i < intArray.size(); i++)
		{
			printf("Array %d: %d \n", i, h_intArray[i]);
		}

		h_intArray.clear();
	}


	struct Flag
	{
		bool toSplit;
		bool old;
		int id;
	};

	template<typename EKey>
	__global__ void HB_SetupTri2Tet(
		GArray<EKey> keys,
		GArray<Flag> flags,
		GArray<bool> toSplit,
		GArray<TopologyModule::Tri2Tet> tri2tet,
		int shift,
		bool bOld)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tri2tet.size()) return;

		Flag flag;
		flag.toSplit = bOld ? toSplit[tId] : false;
		flag.old = bOld;
		flag.id = tId;

		TopologyModule::Tri2Tet tt = tri2tet[tId];
// 
// 		if (bOld && tt[0] != EMPTY && tt[1] != EMPTY && tId == 2)
// 		{
// 			flag.toSplit = true;
// 		}

		keys[shift + tId] = EKey(tt[0], tt[1]);
		flags[shift + tId] = flag;
	}

	template<typename EKey>
	__global__ void HB_UpdateFractureTag(
		GArray<bool> fTag,
		GArray<EKey> keys,
		GArray<Flag> flags)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= keys.size()) return;

		EKey key = keys[tId];
		Flag flag = flags[tId];


// 		if (flag.toSplit == true)
// 		{
// 			printf("Split %d \n", tId);
// 		}
// 		else
// 		{
// 			printf("Not split %d \n", tId);
// 		}

		if (key.isValid() && flag.old == false)
		{
			if (tId < keys.size() - 1 && keys[tId] == keys[tId + 1]/* && flags[tId + 1].old == true*/)
			{
				fTag[flag.id] = flags[tId + 1].toSplit;
// 				if (fTag[flag.id])
// 				{
// 					printf("Surface %d visited at %d \n", flag.id, tId + 1);
// 				}
			}
			if (tId > 0 && keys[tId] == keys[tId - 1]/* && flags[tId - 1].old == true*/)
			{
				fTag[flag.id] = flags[tId - 1].toSplit;
				//printf("Surface %d visited \n", flag.id);
// 				if (fTag[flag.id])
// 				{
// 					printf("Surface %d visited at %d \n", flag.id, tId - 1);
// 				}
			}
		}
	}

	__global__ void HPB_CalculateNeighborTetSize(
		GArray<int> count,
		NeighborList<int> ver2Tet)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= ver2Tet.size()) return;

		count[tId] = ver2Tet.getNeighborSize(tId);
	}

	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::updateTopology()
	{
// 		auto pts = this->m_pSet->getPoints();
// 		Function1Pt::copy(pts, this->currentPosition()->getValue());

		this->m_pSet->setPoints(this->currentPosition()->getValue());

//		return;
		//Error exists
		auto topo = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());

		GArray<TopologyModule::Tetrahedron>& tets = topo->getTetrahedrons();
		GArray<TopologyModule::Tri2Tet>& tri2Tet = topo->getTri2Tet();

		GArray<TopologyModule::Triangle>* tris = topo->getTriangles();

		NeighborList<int>& ver2Tri = topo->getVertex2Triangles();
		NeighborList<int>& ver2Tet = topo->getVer2Tet();

		int verSize = ver2Tri.size();
		int triSize = tris->size();

		GArray<TopologyModule::Tri2Tet> tri2Tet_old;
		tri2Tet_old.resize(tri2Tet.size());
		Function1Pt::copy(tri2Tet_old, tri2Tet);

		NeighborList<Pair<int, int>> pairList;
		pairList.resize(verSize);

		Function1Pt::copy(pairList.getIndex(), ver2Tet.getIndex());
		pairList.getElements().resize(ver2Tet.getElements().size());

		GArray<int> counter;
		counter.resize(verSize);

#ifdef DEBUG_INFO
		printf("Vertex to triangles; \n");
		print(ver2Tri);

		printf("Vertex to tets; \n");
		print(ver2Tet);
#endif
		GArray<int> neiTetNum;
		neiTetNum.resize(ver2Tet.size());

		cuExecute(neiTetNum.size(),
			HPB_CalculateNeighborTetSize,
			neiTetNum,
			ver2Tet);

		Reduction<int> reduce;
		int maxNum = reduce.maximum(neiTetNum.begin(), neiTetNum.size());

		neiTetNum.clear();

		uint pDims = cudaGridSize(verSize, 16);
		uint sharedSize = maxNum * 16 *(sizeof(Pair<int, int>) + sizeof(int));

		HFM_SplitVertex<<<pDims, 16, sharedSize >>>(
			counter,
			pairList,
			this->currentFractureTag()->getValue(),
			ver2Tri,
			ver2Tet,
			tri2Tet,
			tets,
			*tris,
			maxNum);
		cuSynchronize();

		GArray<int> shift;
		shift.resize(counter.size());
		Function1Pt::copy(shift, counter);

		int new_num = thrust::reduce(thrust::device, shift.begin(), shift.begin() + shift.size());
		thrust::exclusive_scan(thrust::device, shift.begin(), shift.begin() + shift.size(), shift.begin());

//		print(shift);

#ifdef DEBUG_INFO
		printf("Before topology update: \n");
		print(tets);
#endif

		GArray<TopologyModule::Tetrahedron> tets_old;
		tets_old.resize(tets.size());
		Function1Pt::copy(tets_old, tets);

		cuExecute(pairList.size(),
			HFM_UpdateTopology,
			tets,
			tets_old,
			counter,
			shift,
			pairList);

#ifdef DEBUG_INFO
		printf("After topology update:  \n");
		print(tets);
#endif

// 		print(tets_old);
// 		print(tets);

		GArray<Coord> position_new;
		position_new.resize(new_num);

		GArray<Coord> velocity_new;
		velocity_new.resize(new_num);

		GArray<Coord> rest_position_new;
		rest_position_new.resize(new_num);

		GArray<Attribute> attribute_new;
		attribute_new.resize(new_num);

		GArray<Coord> yielding_new;
		yielding_new.resize(new_num);

		GArray<Matrix> rotation_new;
		rotation_new.resize(new_num);

		int pNum = this->currentPosition()->getElementCount();
		cuExecute(pNum,
			HFM_UpdateFields,
			rest_position_new,
			this->currentRestPosition()->getValue(),
			position_new,
			this->currentPosition()->getValue(),
			velocity_new,
			this->currentVelocity()->getValue(),
			yielding_new,
			this->currentPrincipleYielding()->getValue(),
			attribute_new,
			this->currentAttribute()->getValue(),
			rotation_new,
			this->currentVertexRotation()->getValue(),
			counter,
			shift);

		//update topology
		topo->setPoints(position_new);
		topo->updateTriangles();

		GArray<TopologyModule::Tri2Tet> tri2Tet_new;
		tri2Tet_new.resize(topo->getTri2Tet().size());
		Function1Pt::copy(tri2Tet_new, topo->getTri2Tet());

		//update fracture tag
		GArray<bool> fractureTag_old;
		fractureTag_old.resize(this->currentFractureTag()->getElementCount());
		Function1Pt::copy(fractureTag_old, this->currentFractureTag()->getValue());

		//print(fractureTag_old);

		this->currentFractureTag()->setElementCount(topo->getTriangles()->size());
		this->currentFractureTag()->getReference()->reset();

		GArray<typename EdgeSet<TDataType>::EKey> keys;
		GArray<Flag> flags;
		int total_tri2tet = tri2Tet_old.size() + tri2Tet_new.size();
		keys.resize(total_tri2tet);
		flags.resize(total_tri2tet);

		cuExecute(tri2Tet_new.size(),
			HB_SetupTri2Tet,
			keys,
			flags,
			this->currentFractureTag()->getValue(),
			tri2Tet_new,
			0,
			false);

		cuExecute(tri2Tet_old.size(),
			HB_SetupTri2Tet,
			keys,
			flags,
			fractureTag_old,
			tri2Tet_old,
			this->currentFractureTag()->getElementCount(),
			true);

		thrust::sort_by_key(thrust::device, keys.begin(), keys.begin() + keys.size(), flags.begin());

		cuExecute(keys.size(),
			HB_UpdateFractureTag,
			this->currentFractureTag()->getValue(),
			keys,
			flags);

//		print(this->currentFractureTag()->getValue());

		//update other fields
		this->currentPosition()->getReference()->resize(new_num);
		this->currentVelocity()->getReference()->resize(new_num);
		this->currentForce()->getReference()->resize(new_num);
		this->currentAttribute()->getReference()->resize(new_num);
		this->currentRestPosition()->getReference()->resize(new_num);
		this->currentPrincipleYielding()->getReference()->resize(new_num);
		this->currentVertexRotation()->getReference()->resize(new_num);

		this->currentPosition()->setValue(position_new);
		this->currentVelocity()->setValue(velocity_new);
		this->currentRestPosition()->setValue(rest_position_new);
		this->currentPrincipleYielding()->setValue(yielding_new);
		this->currentAttribute()->setValue(attribute_new);
		this->currentVertexRotation()->setValue(rotation_new);
		this->currentForce()->getReference()->reset();


		//update rest shape
		this->updateRestShape();
		this->updateVolume();

		fractureTag_old.clear();
		tri2Tet_old.clear();
		tri2Tet_new.clear();
		keys.clear();
		flags.clear();

		pairList.release();
		counter.clear();
		shift.clear();

		position_new.clear();
		velocity_new.clear();
		rest_position_new.clear();
		attribute_new.clear();
		yielding_new.clear();
		tets_old.clear();
		rotation_new.clear();
	}

	template<typename Coord>
	__global__ void HB_InitPlasticYielding(
		GArray<Coord> yieldings)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= yieldings.size()) return;

		yieldings[tId] = Coord(1, 1, 1);
	}

	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::resetStatus()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (!tetSet) return false;
		
		GArray<TopologyModule::Triangle>* tris = tetSet->getTriangles();

		if (tris->size() > 0)
		{
			this->currentFractureTag()->setElementCount(tris->size());
			//this->currentFractureTag()->getReference()->reset();

// 			CArray<bool> host_bool;
// 			host_bool.resize(tris->size());
// 			for (int i = 0; i < tris->size(); i++)
// 			{
// 				host_bool[i] = true;
// 			}
// 
// 			Function1Pt::copy(this->currentFractureTag()->getValue(), host_bool);
// 
// 			host_bool.release();
		}

		int pNum = tetSet->getPoints().size();
		if (pNum > 0)
		{
			this->currentPrincipleYielding()->setElementCount(pNum);
			cuExecute(pNum,
				HB_InitPlasticYielding,
				this->currentPrincipleYielding()->getValue());
		}

		return HyperelasticBody<TDataType>::resetStatus();;
	}


	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::updateStatus()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->getTopologyModule());
		if (!tetSet || !tetSet->isTopologyChanged()) return;

		auto pts = tetSet->getPoints();

		if (pts.size() != this->currentPosition()->getElementCount())
		{
			this->currentPosition()->setElementCount(pts.size());
			this->currentVelocity()->setElementCount(pts.size());
			this->currentForce()->setElementCount(pts.size());

			Function1Pt::copy(this->currentPosition()->getValue(), pts);
			this->currentVelocity()->getReference()->reset();
		}
	}


	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::translate(Coord t)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->translate(t);

		return HyperelasticBody<TDataType>::translate(t);
	}

	template<typename TDataType>
	bool HyperelastoplasticityBody<TDataType>::scale(Real s)
	{
		TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->getTopologyModule())->scale(s);

		return HyperelasticBody<TDataType>::scale(s);
	}


/*	template<typename TDataType>
	void HyperelastoplasticityBody<TDataType>::setElastoplasticitySolver(std::shared_ptr<ElastoplasticityModule<TDataType>> solver)
	{
		auto module = this->getModule("elastoplasticity");
		this->deleteModule(module);

		auto nbrQuery = this->template getModule<NeighborQuery<TDataType>>("neighborhood");

		this->currentPosition()->connect(solver->inPosition());
		this->currentVelocity()->connect(solver->inVelocity());
		nbrQuery->outNeighborhood()->connect(solver->inNeighborhood());
		m_horizon.connect(solver->inHorizon());

		solver->setName("elastoplasticity");
		this->addConstraintModule(solver);
	}*/
}