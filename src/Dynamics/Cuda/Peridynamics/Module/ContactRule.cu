#include "ContactRule.h"

#include "Primitive/Primitive3D.h"
#include "Topology/SparseOctree.h"

#include "CCD/AdditiveCCD.h"
#include "CCD/TightCCD.h"
#include "Vector.h"
#define eps (2e-1)
namespace dyno
{
	IMPLEMENT_TCLASS(ContactRule, TDataType)

		template<typename TDataType>
	ContactRule<TDataType>::ContactRule()
		: ConstraintModule()
	{
		mBroadPhaseCD = std::make_shared<CollisionDetectionBroadPhase<TDataType>>();
	}

	template<typename TDataType>
	ContactRule<TDataType>::~ContactRule()
	{
		firstTri.clear();
		secondTri.clear();
		trueContact.clear();
		trueContactCnt.clear();
		Weight.clear();
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void CR_SetupAABB(
		DArray<AABB> boundingBox,
		DArray<Coord> oldVertices,
		DArray<Coord> newVertices,
		DArray<Triangle> triangles,
		Real epsilon) //This will expend the AABB epsilon time of its length. For the shrinking, please use a negative parameter.
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= boundingBox.size()) return;

		Triangle t = triangles[tId];
		Coord corner1;
		corner1[0] = epsilon; corner1[1] = epsilon; corner1[2] = epsilon;

		Coord v0 = newVertices[t[0]];
		Coord v1 = newVertices[t[1]];
		Coord v2 = newVertices[t[2]];

		Coord s0 = oldVertices[t[0]];
		Coord s1 = oldVertices[t[1]];
		Coord s2 = oldVertices[t[2]];

		AABB box;
		box.v0 = v0;
		box.v1 = v0;

		box.v0 = minimum(box.v0, v1);
		box.v1 = maximum(box.v1, v1);

		box.v0 = minimum(box.v0, v2);
		box.v1 = maximum(box.v1, v2);

		box.v0 = minimum(box.v0, s0);
		box.v1 = maximum(box.v1, s0);

		box.v0 = minimum(box.v0, s1);
		box.v1 = maximum(box.v1, s1);

		box.v0 = minimum(box.v0, s2);
		box.v1 = maximum(box.v1, s2);

		Real len = (box.v0 - box.v1).norm();
		//expend epsilon
		box.v0 += corner1 * len;
		box.v1 += corner1 * len;

		boundingBox[tId] = box;
	}

	template<typename Triangle>
	__device__ bool isTwoTriangleNeighoring(Triangle t0, Triangle t1)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (t0[i] == t1[j])
					return true;
			}
		}

		return false;
	}


	template<typename Coord, typename Triangle, typename Real>
	__global__ void CR_Calculate_Timestep(
		DArray<Real> timestep,
		DArray<Coord> vertexOld,
		DArray<Coord> vertexNew,
		DArray<Triangle> triangles,
		DArrayList<int> contactList,
		Real thickness,
		Real collisionRefactor,
		DArrayList<int> trueContact,
		DArray<uint> trueContactCnt)
	{

		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= timestep.size()) return;

		Triangle t_i = triangles[tId];
		List<int>& list_i = contactList[tId]; // vertex's in triangle list
		int nbSize = list_i.size();
		Real time = Real(1.0);

		trueContactCnt[tId] = 0;


		TTriangle3D<Real> tri_old_i(vertexOld[t_i[0]], vertexOld[t_i[1]], vertexOld[t_i[2]]);
		TTriangle3D<Real> tri_new_i(vertexNew[t_i[0]], vertexNew[t_i[1]], vertexNew[t_i[2]]);

		List<int>& tContact = trueContact[tId];

		for (int n = 0; n < nbSize; n++)
		{
			int j = list_i[n];
			Triangle t_j = triangles[j];

			if (!isTwoTriangleNeighoring(t_i, t_j))
			{
				TTriangle3D<Real> tri_old_j(vertexOld[t_j[0]], vertexOld[t_j[1]], vertexOld[t_j[2]]);
				TTriangle3D<Real> tri_new_j(vertexNew[t_j[0]], vertexNew[t_j[1]], vertexNew[t_j[2]]);

				Real toi = Real(1.0);

				auto ccdPhase = AdditiveCCD<Real>(thickness, collisionRefactor, 0.95);
				bool collided = ccdPhase.TriangleCCD(tri_old_i, tri_new_i, tri_old_j, tri_new_j, toi);
				
				time = collided ? minimum(time, toi) : time;
				if (collided) {
					for (int index = 0; index < 3; ++index) {
						tContact.insert(j);
						trueContactCnt[tId] += 1;
					}
				}
			}
		}

		timestep[tId] = time;
		//if(time<1.0 && time>EPSILON)printf("contact %f\n",time);
		
	}


	template<typename Real>
	__global__ void CR_Calculate_Timestep(
		DArray<Real> timestepOfVertex,		//timestep for each vertex
		DArray<Real> timestepOfTriangle,
		DArrayList<int> ver2tri)
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= timestepOfVertex.size()) return;

		List<int>& list_i = ver2tri[vId];
		int nbSize = list_i.size();

		Real time = Real(1);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];

			time = minimum(time, timestepOfTriangle[j]);
		}

		timestepOfVertex[vId] = time;

	}



	template<typename Real, typename Coord>
	__global__ void CR_Update_Vertex(
		DArray<Coord> vertexNew,
		DArray<Coord> vertexOld,
		DArray<Real> timestep)	//timestep for each triangle
	{
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= timestep.size()) return;

		Coord vOld = vertexOld[vId];
		Coord vNew = vertexNew[vId];

		Real timestep_i = timestep[vId];

		vertexNew[vId] = vOld + (vNew - vOld) * timestep_i;

	}


template<typename Coord, typename Real>
__global__ void CR_Init_Force_Weight(
	DArray<Coord> force,
	DArray<Real> Weight) {
	int vId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (vId >= force.size()) return;

	force[vId] = Coord(0); 
	Weight[vId] = (Real)0.0;

}


template<typename Coord, typename Triangle>
__global__ void CR_Update_Distance(
	DArrayList<int> ContactList,
	DArrayList<Coord> firstTri,
	DArrayList<Coord> secondTri,
	DArray<Triangle> triangles,
	DArray<Coord> vertexNew) {
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= ContactList.size()) return;

	List<int>& list_i = ContactList[tId];
	Triangle t_i = triangles[tId];
	TTriangle3D<Real> tri_i(vertexNew[t_i[0]], vertexNew[t_i[1]], vertexNew[t_i[2]]);
	int nbSize = list_i.size();
	auto ccdPhase = AdditiveCCD<Real>();
	List<Coord>& fi = firstTri[tId];
	List<Coord>& se = secondTri[tId];
	for (int ne = 0; ne < nbSize; ++ne) {
		int j = list_i[ne];
		Triangle t_j = triangles[j];
		TTriangle3D<Real> tri_j(vertexNew[t_j[0]], vertexNew[t_j[1]], vertexNew[t_j[2]]);
		Vector<Real, 3> firstT, secondT;
		ccdPhase.projectClosePoint(tri_i, tri_j, firstT, secondT);
		Coord f, s;
		f[0] = firstT[0], f[1] = firstT[1], f[2] = firstT[2];
		s[0] = secondT[0], s[1] = secondT[1], s[2] = secondT[2];

		fi.insert(f);
		se.insert(s);
	}
}

template<typename Coord, typename Real, typename Triangle>
__global__ void CR_UpDate_Contact_Force(
	DArrayList<int> ContactList,
	DArrayList<Coord> firstT,
	DArrayList<Coord> secondT,
	DArray<Coord> contactForce,
	Real d,
	Real xi,
	DArray<Triangle> triangles,
	DArray<Coord> vertexNew,
	DArray<Real> Weight) {
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tId >= firstT.size()) return;

	auto myNorm = [](Coord v0, Coord v1) {
		return sqrt(pow(v0[0] - v1[0], 2) + pow(v0[1] - v1[1], 2) + pow(v0[2] - v1[2], 2));
	};
	auto calcN = [&](Coord v, const Triangle& t_i, const Triangle& t_j) {
		Coord e1 = vertexNew[t_i[0]] / d - vertexNew[t_i[1]] / d;
		Coord e2 = vertexNew[t_i[0]] / d - vertexNew[t_i[2]] / d;
		Coord n = e1.cross(e2);
		Real nL = sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));
		n /= nL;
		Coord vj_b = v - 1.0 / 3.0 * (vertexNew[t_j[0]] + vertexNew[t_j[1]] + vertexNew[t_j[2]]);
		Real Dot = n[0] * vj_b[0] + n[1] * vj_b[1] + n[2] * vj_b[2];
		n *= Dot <= 0.0 ? -1.0 : 1.0;
		return n;
	};

	auto refactor_n = [&](Real n) {
		if (n >= (Real)((1.0 + 2.0 * eps) * xi))
			return n;
		if (n <= xi)
		{
			Real n_re = Real((1.0 + 1.0 * eps) * xi);
			return n_re;
		}
		Real a = Real(1.0 / (4.0 * eps * xi));
		Real n_re =Real( a * pow(n - xi, 2) + (1.0 + 1.0 * eps) * xi);
		return n_re;
	};

	auto contactforce = [&](Coord v1, Coord v2, const Triangle& t_i, const Triangle& t_j, Coord vi_c, Coord vj_c) {
		Coord force = v1 * 0.0;
		Real n = myNorm(v1, v2);
		if (n <= (Real)(1.0+2*eps) * xi) {
			for (int index_tri = 0; index_tri < 3; ++index_tri) {
				atomicAdd(&Weight[t_i[index_tri]], vi_c[index_tri]);
			}
		}
		n = refactor_n(n);
		Coord N = (v1 - v2) / myNorm(v1, v2);
		force = (pow((n -  xi) - d, 2) / (n -  xi) + log((n - xi) / d) * 2 * (n -  xi - d)) * N;
		return force;
	};

	auto contactforceN = [&](Coord v1, Coord v2, Coord N, const Triangle& t_i, const Triangle& t_j, Coord vi_c, Coord vj_c) {
		Coord force = v1 * 0.0;
		Real n = myNorm(v1, v2);
		if (n <= (Real)(1.0 + 2*eps) * xi) {
			for (int index_tri = 0; index_tri < 3; ++index_tri) {
				atomicAdd(&Weight[t_i[index_tri]], vi_c[index_tri]);
			}
		}
		n = refactor_n(n);
		force = (pow(n - xi - d, 2) / (n - xi) + log((n - xi) / d) * 2.0 * (n - xi - d)) * N;
		return force;
	};

		List<int>& list_i = ContactList[tId];
		List<Coord>& tri_coordinate_i = firstT[tId];
		List<Coord>& tri_coordinate_j = secondT[tId];
		Triangle t_i = triangles[tId];
		int nbSize = list_i.size();

		for (int ne = 0; ne < nbSize; ne++) {

			int j = list_i[ne];
			Triangle t_j = triangles[j];
			Coord vi_N = tri_coordinate_i[ne][0] * vertexNew[t_i[0]] + tri_coordinate_i[ne][1] * vertexNew[t_i[1]] + tri_coordinate_i[ne][2] * vertexNew[t_i[2]];
			Coord vj_N = tri_coordinate_j[ne][0] * vertexNew[t_j[0]] + tri_coordinate_j[ne][1] * vertexNew[t_j[1]] + tri_coordinate_j[ne][2] * vertexNew[t_j[2]];
			Real disN = myNorm(vi_N, vj_N);
			
			if (disN > d)
				break;
			
			vi_N = tri_coordinate_i[ne][0] * vertexNew[t_i[0]] + tri_coordinate_i[ne][1] * vertexNew[t_i[1]] + tri_coordinate_i[ne][2] * vertexNew[t_i[2]];
			vj_N = tri_coordinate_j[ne][0] * vertexNew[t_j[0]] + tri_coordinate_j[ne][1] * vertexNew[t_j[1]] + tri_coordinate_j[ne][2] * vertexNew[t_j[2]];
			Coord force;
			if (myNorm(vi_N, vj_N) >= EPSILON)
				force = contactforce(vi_N, vj_N,t_i,t_j, tri_coordinate_i[ne], tri_coordinate_j[ne]);
			else
				force = contactforceN(vi_N, vj_N, calcN(vi_N,t_i,t_j), t_i, t_j, tri_coordinate_i[ne], tri_coordinate_j[ne]);

			for (int index_tri = 0; index_tri < 3; ++index_tri) {
				for (int index_Coord = 0; index_Coord < 3; ++index_Coord) {
					atomicAdd(&contactForce[t_i[index_tri]][index_Coord], force[index_Coord] * (tri_coordinate_i[ne][index_tri]));
				}
			}
		}

	}

	template<typename Coord, typename Real>
	__global__ void CR_Weighted_Force(
		DArray<Real> Weight,
		DArray<Coord> force,
		DArray<Coord> Position,
		DArray<Coord> OldPosition,
		Real timestep) {
		Real t = timestep;
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= Weight.size()) return;
		auto myNorm = [](Coord v0) {
			return sqrt(pow(v0[0], 2) + pow(v0[1], 2) + pow(v0[2], 2));
		};
		
		if (myNorm(force[vId]) <= EPSILON)
			return;
		if (myNorm(force[vId]) > 1e16)
			force[vId] *= 0.0;
		
		Position[vId] += (force[vId] + OldPosition[vId]-Position[vId])* t;
	}

	__global__ void CR_Cnt(
		DArrayList<int>cList,
		DArray<uint> cnt ) {
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= cnt.size()) return;
		cnt[tId] = cList[tId].size();
	}

	template<typename Coord, typename Real>
	__global__ void CR_Force_Magnitude(
		DArray<Real> force_m,
		DArray<Coord> force,
		DArray<Real> weight,
		DArray<Coord> oldPosition,
		DArray<Coord> newPosition) {
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= force.size()) return;
		auto myNorm = [](Coord v0) {
			return sqrt(pow(v0[0], 2) + pow(v0[1], 2) + pow(v0[2], 2));
		};
		force_m[vId] = myNorm(force[vId]);
	}	


	template<typename Coord, typename Real>
	__global__ void CR_Cancle_Position(
		DArray<Real> Weight,
		DArray<Coord> Position,
		DArray<Coord> OldPosition) {
		int vId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (vId >= Weight.size()) return;
		if(Weight[vId] - EPSILON>=0.0)
			Position[vId] = OldPosition[vId];
	}

	template<typename TDataType>
	void ContactRule<TDataType>::initCCDBroadPhase()
	{
		auto topo = this->inTriangularMesh()->getDataPtr();
		int vNum = this->inOldPosition()->size();
		auto& indices = topo->getTriangles();
		auto& ver2tri = topo->getVertex2Triangles();

		int tNum = indices.size();

		DArray<AABB> aabb(tNum);
		cuExecute(tNum,
			CR_SetupAABB,
			aabb,
			this->inOldPosition()->getData(),
			this->inNewPosition()->getData(),
			indices,
			this->inXi()->getData() * this->inUnit()->getData());

		mBroadPhaseCD->varGridSizeLimit()->setValue(this->inUnit()->getData());
		mBroadPhaseCD->inSource()->assign(aabb);
		mBroadPhaseCD->inTarget()->assign(aabb);
		mBroadPhaseCD->update();
		auto& cList = mBroadPhaseCD->outContactList()->getData();
		
		if (this->outContactForce()->isEmpty())
			this->outContactForce()->allocate();
		this->outContactForce()->getData().resize(vNum);

		if (this->outWeight()->isEmpty())
			this->outWeight()->allocate();
		this->outWeight()->getData().resize(vNum);

		cuExecute(vNum,
			CR_Init_Force_Weight,
			this->outContactForce()->getData(),
			this->outWeight()->getData());
		
		this->trueContactCnt.resize(tNum);
		
		aabb.clear();
	}

	

	template<typename TDataType>
	void ContactRule<TDataType>::constrain()
	{
	
		int vNum = this->inOldPosition()->size();
		auto topo = this->inTriangularMesh()->getDataPtr();
		auto& indices = topo->getTriangles();
		int tNum = indices.size();
		auto& ver2tri = topo->getVertex2Triangles();

		auto& contactForce = this->outContactForce()->getData();
		if (contactForce.size() != vNum) {
			contactForce.resize(vNum);
			if (this->outWeight()->isEmpty())
				this->outWeight()->allocate();
			this->outWeight()->getData().resize(vNum);
			cuExecute(vNum,
				CR_Init_Force_Weight,
				this->outContactForce()->getData(),
				this->outWeight()->getData());
		}
	
		auto& cList = mBroadPhaseCD->outContactList()->getData();

		cuExecute(tNum,
			CR_Cnt,
			cList,
			this->trueContactCnt);
		cuSynchronize();

		this->trueContact.resize(this->trueContactCnt);
		
		DArray<Real> steplengthOfVertex(vNum);
		DArray<Real> steplengthOfTriangle(tNum);
		Real d_hat = Real((1.0+15*eps) * this->inXi()->getData() * this->inUnit()->getData());

		cuExecute(tNum,
			CR_Calculate_Timestep,
			steplengthOfTriangle,
			this->inOldPosition()->getData(),
			this->inNewPosition()->getData(),
			indices,
			cList,
			Real(this->inXi()->getData() * this->inUnit()->getData()),
			this->inS()->getData(),
			this->trueContact,
			this->trueContactCnt
			);
		
		cuExecute(vNum,
			CR_Calculate_Timestep,
			steplengthOfVertex,
			steplengthOfTriangle,
			ver2tri);
		cuSynchronize();
	
		cuExecute(vNum,
			CR_Update_Vertex,
			this->inNewPosition()->getData(),
			this->inOldPosition()->getData(),
			steplengthOfVertex);

		cuSynchronize();
		DArray<Coord> msBof;
		msBof.resize(vNum);
		msBof.assign(this->inNewPosition()->getData());
		int ite = 0;
		Real maxWeight = 0.0;
		Real maxforce = 0.1;
		Reduction<Real> reduce;
		this->weight = -1;
		bool status = true;
		while(ite < this->ContactMaxIte && maxforce >= 1e-2){


			this->secondTri.resize(this->trueContactCnt);
			this->firstTri.resize(this->trueContactCnt);
		
			cuExecute(tNum,
				CR_Update_Distance,
				this->trueContact,
				this->firstTri,
				this->secondTri,
				indices,
				this->inNewPosition()->getData());

			cuExecute(vNum,
				CR_Init_Force_Weight,
				this->outContactForce()->getData(),
				this->outWeight()->getData());

			cuExecute(tNum,
				CR_UpDate_Contact_Force,
				this->trueContact,
				this->firstTri,
				this->secondTri,
				this->outContactForce()->getData(),
				d_hat,
				(Real)(this->inXi()->getData() * this->inUnit()->getData()),
				indices,
				this->inNewPosition()->getData(),
				this->outWeight()->getData());

			cuSynchronize();
			DArray<Real> force_m;
			force_m.resize(vNum);
			cuExecute(vNum,
				CR_Force_Magnitude,
				force_m,
				this->outContactForce()->getData(),
				this->outWeight()->getData(),
				msBof,
				this->inNewPosition()->getData());
			cuSynchronize();
			
			Real m = reduce.maximum(this->outWeight()->getData().begin(), this->outWeight()->getData().size());
			maxWeight = maximum(m, maxWeight);
			maxforce = reduce.maximum(force_m.begin(), force_m.size());
			if (m <= EPSILON || maxforce <=1e-2)
			{
				printf("============== contact Rule: ite = %d ================\n", ite);
				status = false;
				break;
			}
			
			if (maxforce > 1e16)
				maxforce = 1e16;
			
			Real u_xi = Real(this->inUnit()->getData()* this->inXi()->getData());
			Real max_f = pow(eps * u_xi - d_hat, 2) / (eps * u_xi) + log((eps * u_xi) / d_hat) * 2.0 * (eps * u_xi - d_hat);
			this->weight = max_f;
			Real timestep  = minimum(Real( eps * u_xi/((maxWeight*max_f)+0.01)), Real(eps* u_xi / (maxforce)));
			//printf("maxWeight ,maxforce,timestep %f %f %f\n", maxWeight, maxforce,timestep);
			cuExecute(vNum,
				CR_Weighted_Force,
				this->outWeight()->getData(),
				this->outContactForce()->getData(),
				this->inNewPosition()->getData(),
				msBof,
				timestep);

			++ite;
			cuSynchronize();
			force_m.clear();
		}
		
		msBof.clear();
		steplengthOfVertex.clear();
		steplengthOfTriangle.clear();
		if(status)
			printf("========= contact Rule clamped with ite = %d =========\n", ite);
	}

	DEFINE_CLASS(ContactRule);
}