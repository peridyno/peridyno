#include "DynamicNeighborTriangleQuery.h"
#include "Primitive/Primitive3D.h"

#include "Collision/CollisionDetectionBroadPhase.h"

namespace dyno
{
	IMPLEMENT_TCLASS(DynamicNeighborTriangleQuery, TDataType)

		template<typename TDataType>
	DynamicNeighborTriangleQuery<TDataType>::DynamicNeighborTriangleQuery()
		: ComputeModule()
	{
		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	DynamicNeighborTriangleQuery<TDataType>::~DynamicNeighborTriangleQuery()
	{
		mQueryAABB.clear();
		mQueriedAABB.clear();
	}

	template<typename Coord>
	__global__ void NTQ_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> position,
		Real radius,
		DArray<Coord> vel,
		Coord g,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		AABB box;
		Coord p_end = position[pId];
		Coord v = vel[pId] - g * dt;
		Coord p_strat = position[pId] - v * dt;
		Real max_x = maximum(p_strat.x, p_end.x);
		Real max_y = maximum(p_strat.y, p_end.y);
		Real max_z = maximum(p_strat.z, p_end.z);
		Real min_x = minimum(p_strat.x, p_end.x);
		Real min_y = minimum(p_strat.y, p_end.y);
		Real min_z = minimum(p_strat.z, p_end.z);
		box.v0 = Coord(min_x, min_y, min_z) - radius;
		box.v1 = Coord(max_x, max_y, max_z) + radius;
		//printf("p.x = %f, p.y = %f, p.z = %f\n", position[pId].x, position[pId].y, position[pId].z);
		//printf("pp.x = %f, pp.y = %f, pp.z = %f\n", (position[pId] - v * dt).x, (position[pId] - v * dt).y, (position[pId] - v * dt).z);
		//printf("box.v0.x = %f, box.v0.y = %f, box.v0.z = %f\n", box.v0.x, box.v0.y, box.v0.z);
		//printf("box.v1.x = %f, box.v1.y = %f, box.v1.z = %f\n", box.v1.x, box.v1.y, box.v1.z);


		boundingBox[pId] = box;
	}

	template<typename Coord>
	__global__ void NTQ_SetupTriAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> vertex,
		DArray<Topology::Triangle> tIndex,
		DArray<Coord> prevertex,
		Real radius)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		AABB box;
		Topology::Triangle index = tIndex[tId];

		Coord v0 = vertex[index[0]];
		Coord v1 = vertex[index[1]];
		Coord v2 = vertex[index[2]];
		Coord p0 = prevertex[index[0]];
		Coord p1 = prevertex[index[1]];
		Coord p2 = prevertex[index[2]];
		Real max_x = maximum(maximum(p0.x, maximum(p1.x, p2.x)), maximum(v0.x, maximum(v1.x, v2.x)));
		Real max_y = maximum(maximum(p0.y, maximum(p1.y, p2.y)), maximum(v0.y, maximum(v1.y, v2.y)));
		Real max_z = maximum(maximum(p0.z, maximum(p1.z, p2.z)), maximum(v0.z, maximum(v1.z, v2.z)));
		Real min_x = minimum(minimum(p0.x, minimum(p1.x, p2.x)), minimum(v0.x, minimum(v1.x, v2.x)));
		Real min_y = minimum(minimum(p0.y, minimum(p1.y, p2.y)), minimum(v0.y, minimum(v1.y, v2.y)));
		Real min_z = minimum(minimum(p0.z, minimum(p1.z, p2.z)), minimum(v0.z, minimum(v1.z, v2.z)));
		box.v0 = Coord(min_x, min_y, min_z) - radius;
		box.v1 = Coord(max_x, max_y, max_z) + radius;

		//printf("Triangle: box.v0.x = %f, box.v0.y = %f, box.v0.z = %f\n", box.v0.x, box.v0.y, box.v0.z);
		//printf("Triangle: box.v1.x = %f, box.v1.y = %f, box.v1.z = %f\n", box.v1.x, box.v1.y, box.v1.z);

		boundingBox[tId] = box;
	}

	template<typename Coord>
	__global__ void NTQ_Narrow_Count(
		DArrayList<int> nbr,
		DArray<Coord> position,
		DArray<Coord> vertex,
		DArray<Topology::Triangle> triangles,
		DArray<Coord> prevertex,
		DArray<uint> count,
		Real radius,
		DArray<Coord> vel,
		Coord g,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		uint cnt = 0;

		List<int>& nbrIds_i = nbr[pId];
		int nbSize = nbrIds_i.size();
		//printf("nbSize: %d\n", nbSize);
		Coord v = vel[pId] - g * dt;
		Point3D p3d_end(position[pId]);
		Point3D p3d_start(position[pId] - v * dt);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Triangle3D t3d_end(vertex[triangles[j][0]], vertex[triangles[j][1]], vertex[triangles[j][2]]);
			Triangle3D t3d_start(prevertex[triangles[j][0]], prevertex[triangles[j][1]], prevertex[triangles[j][2]]);
			Real p_dis_start = p3d_start.distance(t3d_start);
			Real p_dis_end = p3d_end.distance(t3d_end);
			radius = maximum(radius, v.norm() * dt);
			if (glm::abs(p_dis_end) < radius || p_dis_start * p_dis_end < EPSILON)
			{
				cnt++;
			}
		}
		count[pId] = cnt;
		//printf("cnt: %d\n", cnt);
	}

	template<typename Coord>
	__global__ void NTQ_Narrow_Set(
		DArrayList<int> nbr,
		DArrayList<int> nbr_out,
		DArray<Coord> position,
		DArray<Coord> vertex,
		DArray<Topology::Triangle> triangles,
		DArray<Coord> prevertex,
		Real radius,
		DArray<Coord> vel,
		Coord g,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;

		List<int>& nbrIds_i = nbr[pId];
		int nbSize = nbrIds_i.size();
		List<int>& nbrOutIds_i = nbr_out[pId];
		Coord v = vel[pId] - g * dt;
		Point3D p3d_end(position[pId]);
		Point3D p3d_start(position[pId] - v * dt);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = nbrIds_i[ne];
			Triangle3D t3d_end(vertex[triangles[j][0]], vertex[triangles[j][1]], vertex[triangles[j][2]]);
			Triangle3D t3d_start(prevertex[triangles[j][0]], prevertex[triangles[j][1]], prevertex[triangles[j][2]]);
			Real p_dis_start = p3d_start.distance(t3d_start);
			Real p_dis_end = p3d_end.distance(t3d_end);
			radius = maximum(radius, v.norm() * dt);
			if (glm::abs(p_dis_end) < radius || p_dis_start * p_dis_end < EPSILON)
			{
				nbrOutIds_i.insert(j);
				//printf("pid:%d, tri:%d\n", pId, j);
			}
		}
	}

	template<typename TDataType>
	void DynamicNeighborTriangleQuery<TDataType>::compute()
	{
		int pNum = this->inPosition()->size();
		if (pNum == 0) return;

		if (mQueryAABB.size() != pNum) {
			mQueryAABB.resize(pNum);
		}

		auto ts = this->inTriangleSet()->constDataPtr();
		auto& triVertex = ts->getPoints();
		auto& triIndex = ts->triangleIndices();

		int tNum = triIndex.size();
		if (tNum == 0) return;
		if (mQueriedAABB.size() != tNum) {
			mQueriedAABB.resize(tNum);
		}
		Real dt = this->inTimeStep()->getData();
		Coord  g = this->varGravity()->getValue();
		auto& vels = this->inVelocity()->getData();
		auto& poss = this->inPosition()->getData();

		cuExecute(pNum,
			NTQ_SetupAABB,
			mQueryAABB,
			this->inPosition()->getData(),
			this->inRadius()->getData(),
			vels,
			g,
			dt);

		cuExecute(tNum,
			NTQ_SetupTriAABB,
			mQueriedAABB,
			triVertex,
			triIndex,
			this->inPreTriPosition()->getData(),
			this->inRadius()->getData());

		Real radius = this->inRadius()->getData();

		mBroadPhaseCD->varGridSizeLimit()->setValue(radius);
		mBroadPhaseCD->inSource()->assign(mQueryAABB);
		mBroadPhaseCD->inTarget()->assign(mQueriedAABB);

		auto type = this->varSpatial()->getDataPtr()->currentKey();

		switch (type)
		{
		case Spatial::BVH:
			mBroadPhaseCD->varAccelerationStructure()->getDataPtr()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::BVH);
		case Spatial::OCTREE:
			mBroadPhaseCD->varAccelerationStructure()->getDataPtr()->setCurrentKey(CollisionDetectionBroadPhase<TDataType>::Octree);
		default:
			break;
		}

		mBroadPhaseCD->update();

		auto& nbr = mBroadPhaseCD->outContactList()->getData();

		if (this->outNeighborIds()->size() != pNum)
		{
			this->outNeighborIds()->allocate();
			DArray<uint> nbrNum;
			nbrNum.resize(pNum);
			nbrNum.reset();
			auto& nbrIds = this->outNeighborIds()->getData();
			nbrIds.resize(nbrNum);
			nbrNum.clear();

			//this->outNeighborIds()->getData().resize(p_num);
		}
		auto& nbrIds = this->outNeighborIds()->getData();

		//new
		DArray<uint> nbrNum;
		nbrNum.resize(pNum);
		nbrNum.reset();
		int sum1 = nbr.elementSize();
		//printf("sum1: %d\n", sum1);
		if (sum1 > 0)
		{
			//printf("one %d %d\n", nbrNum.size(), p_num);
			cuExecute(pNum,
				NTQ_Narrow_Count,
				nbr,
				poss,
				triVertex,
				triIndex,
				this->inPreTriPosition()->getData(),
				nbrNum,
				this->inRadius()->getData(),
				vels,
				g,
				dt);//0.9

			nbrIds.resize(nbrNum);

			int sum = mReduce.accumulate(nbrNum.begin(), nbrNum.size());
			if (sum > 0)
			{
				cuExecute(pNum,
					NTQ_Narrow_Set,
					nbr,
					nbrIds,
					poss,
					triVertex,
					triIndex,
					this->inPreTriPosition()->getData(),
					this->inRadius()->getData(),
					vels,
					g,
					dt);//0.9

				nbrNum.clear();
			}
		}
	}

	DEFINE_CLASS(DynamicNeighborTriangleQuery);
}