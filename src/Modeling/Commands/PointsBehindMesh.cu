#include "PointsBehindMesh.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

#include <thrust/sort.h>
namespace dyno
{
	template<typename TDataType>
	PointsBehindMesh<TDataType>::PointsBehindMesh()
		: Node()
	{
		this->statePointSet()->setDataPtr(std::make_shared<PointSet<TDataType>>());
		this->statePointSet()->promoteOuput();

		this->statePlane()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->statePosition()->allocate();
		this->statePosition()->promoteOuput();

		this->statePointNormal()->allocate();
		this->statePointNormal()->promoteOuput();

		this->statePointBelongTriangleIndex()->allocate();


		//auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		//tsRender->setColor(Color(0.1f, 0.12f, 0.25f));
		//tsRender->varAlpha()->setValue(0.5);
		//tsRender->setVisible(true);
		//this->statePlane()->connect(tsRender->inTriangleSet());
		//this->graphicsPipeline()->pushModule(tsRender);

		auto esRender = std::make_shared<GLWireframeVisualModule>();
		esRender->varBaseColor()->setValue(Color(0, 0, 0));
		this->statePlane()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->setColor(Color(0, 0.5, 1));
		ptRender->varPointSize()->setValue(0.005f);
		ptRender->setColorMapMode(GLPointVisualModule::PER_VERTEX_SHADER);
		this->statePointSet()->connect(ptRender->inPointSet());
		this->graphicsPipeline()->pushModule(ptRender);

		m_NeighborPointQuery = std::make_shared<NeighborPointQuery<TDataType>>();
		this->statePosition()->connect(m_NeighborPointQuery->inPosition());
	};


	template<typename Coord>
	__device__ Real WhetherPointInsideTriangle(Coord t_a, Coord t_b, Coord t_c, Coord point_o) {

		/*@Brief : Caculate triangle arear.
		*/

		Coord ab = t_b - t_a;
		Coord ac = t_c - t_a;
		Real abc = (ab.cross(ac)).norm() * 0.5;

		Coord ao = point_o - t_a;
		Coord bo = point_o - t_b;
		Coord bc = t_c - t_b;

		Real abo = (ab.cross(ao)).norm() * 0.5;
		Real aco = (ac.cross(ao)).norm() * 0.5;
		Real bco = (bc.cross(bo)).norm() * 0.5;

		return ((abo + aco + bco) - abc);
	}



	template<typename Coord, typename Triangle>
	__global__ void CalculateSquareFromTriangle(
		DArray<Coord> square_a,
		DArray<Coord> square_b,
		DArray<Coord> square_c,
		DArray<Coord> square_d,
		DArray<Triangle> triangles,
		DArray<Coord> vertices
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Triangle t = triangles[tId];
		const Coord& v0 = vertices[t[0]];
		const Coord& v1 = vertices[t[1]];
		const Coord& v2 = vertices[t[2]];

		Real lv01 = (v0 - v1).norm();
		Real lv12 = (v1 - v2).norm();
		Real lv02 = (v0 - v2).norm();

		Coord C(0.0f);
		if ((lv01 >= lv12) && (lv01 >= lv02))
		{
			square_a[tId] = v0;
			square_b[tId] = v1;
			C = v2;
		}
		else if ((lv12 >= lv01) && (lv12 >= lv02))
		{
			square_a[tId] = v1;
			square_b[tId] = v2;
			C = v0;
		}
		else if ((lv02 >= lv01) && (lv02 >= lv12))
		{
			square_a[tId] = v0;
			square_b[tId] = v2;
			C = v1;
		}

		Coord AB = square_b[tId] - square_a[tId];
		Coord n_AB = AB / AB.norm();
		Coord AC = C - square_a[tId];
		Coord proj_O = square_a[tId] + (AC.dot(n_AB)) * (n_AB);
		Coord trans_vec = C - proj_O;

		square_c[tId] = square_a[tId] + trans_vec;
		square_d[tId] = square_b[tId] + trans_vec;
	}


	template<typename Coord, typename Triangle>
	__global__ void CalculateNewTriangle(
		DArray<Triangle> NewTriangles,
		DArray<Coord> square_a,
		DArray<Coord> square_b,
		DArray<Coord> square_c,
		DArray<Coord> square_d,
		DArray<Coord> vertices
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= square_a.size()) return;

		int v_id = tId * 4;

		vertices[v_id] = square_a[tId];
		vertices[v_id + 1] = square_b[tId];
		vertices[v_id + 2] = square_c[tId];
		vertices[v_id + 3] = square_d[tId];

		int t_id = tId * 2;

		Triangle& t0 = NewTriangles[t_id];
		t0[0] = v_id;
		t0[1] = v_id + 1;
		t0[2] = v_id + 2;

		Triangle& t1 = NewTriangles[t_id + 1];

		t1[0] = v_id + 1;
		t1[1] = v_id + 2;
		t1[2] = v_id + 3;

	}


	template<typename Coord, typename Triangle, typename Real>
	__global__ void CalculateTriangleBasicVector(
		DArray<Triangle> triangles,
		DArray<Coord> vertices,
		DArray<Coord> basic_x,
		DArray<Coord> basic_y,
		DArray<Coord> basic_z,
		DArray<Coord> square_a,
		DArray<Coord> square_b,
		DArray<Coord> square_c,
		DArray<Coord> square_d,
		Real dx
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= triangles.size()) return;

		Coord& n_x = basic_x[tId];
		Coord& n_y = basic_y[tId];
		Coord& n_z = basic_z[tId];

		n_x = (square_b[tId] - square_a[tId]) / (square_b[tId] - square_a[tId]).norm();
		n_y = (square_c[tId] - square_a[tId]) / (square_c[tId] - square_a[tId]).norm();

		n_z = n_x.cross(n_y);
		n_z = n_z / n_z.norm();
	}


	template<typename Coord, typename Real, typename Triangle>
	__global__ void CalculatePointSizeInSquare(
		DArray<int> PointSize,
		DArray<Triangle> triangles,
		DArray<Coord> vertices,
		DArray<Coord> basic_x,
		DArray<Coord> basic_y,
		DArray<Coord> basic_z,
		DArray<Coord> square_a,
		DArray<Coord> square_b,
		DArray<Coord> square_c,
		DArray<Coord> square_d,
		Real dx
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= PointSize.size()) return;

		int& size = PointSize[tId];
		PointSize[tId] = 0;

		int num_x = (int)((square_a[tId] - square_c[tId]).norm() / dx);
		int num_y = (int)((square_a[tId] - square_b[tId]).norm() / dx);

		Triangle t = triangles[tId];
		const Coord& v0 = vertices[t[0]];
		const Coord& v1 = vertices[t[1]];
		const Coord& v2 = vertices[t[2]];

		for (int j = 0; j <= num_y + 1; j++)
		{
			for (int i = 0; i <= num_x + 1; i++)
			{
				int count = i + j * num_x;

				//if (count < size) {
				Coord candi = square_a[tId] + (Real)(j)*dx * basic_x[tId] + (Real)(i)*dx * basic_y[tId];
				Real value = WhetherPointInsideTriangle(v0, v1, v2, candi);
				if (value < 100000000 * dx * dx * EPSILON) {
					size++;
				}

			}
		}
	}


	template<typename Matrix, typename Coord, typename Real, typename Triangle>
	__global__ void CalculatePointsOnPlane(
		DArray<Matrix> Rotation,
		DArray<Coord> Normal,
		DArray<Triangle> triangles,
		Real value
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= Normal.size()) return;

	}


	template<typename Coord, typename Real, typename Triangle>
	__global__ void CalculatePointInSquare(
		DArray<Coord> Points,
		DArray<int> PointSize,
		DArray<int> tIndex,
		DArray<Coord> PointNormal,
		DArray<int> PointOfTriangle,
		DArray<Triangle> triangles,
		DArray<Coord> vertices,
		DArray<Coord> basic_x,
		DArray<Coord> basic_y,
		DArray<Coord> PlaneNormal,
		DArray<Coord> square_a,
		DArray<Coord> square_b,
		DArray<Coord> square_c,
		DArray<Coord> square_d,
		Real dx
	)
	{

		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= PointSize.size()) return;


		int num_x = (int)((square_a[tId] - square_c[tId]).norm() / dx);
		int num_y = (int)((square_a[tId] - square_b[tId]).norm() / dx);
		int& begin = tIndex[tId];
		int& size = PointSize[tId];


		Triangle t = triangles[tId];
		const Coord& v0 = vertices[t[0]];
		const Coord& v1 = vertices[t[1]];
		const Coord& v2 = vertices[t[2]];

		int counter = 0;

		for (int j = 0; j <= num_y + 1; j++)
		{
			for (int i = 0; i <= num_x + 1; i++)
			{
				//int count = i + j * num_x;
				if (counter < size) {

					Coord candi = square_a[tId] + (Real)(j)*dx * basic_x[tId] + (Real)(i)*dx * basic_y[tId];
					Real value = WhetherPointInsideTriangle(v0, v1, v2, candi);

					if (value < 100000000 * dx * dx * EPSILON)
					{
						Points[begin + counter] = candi;
						PointOfTriangle[begin + counter] = tId;
						PointNormal[begin + counter] = -1.0 * PlaneNormal[tId];
						counter++;
					}

				}
			}
		}
	}

	template<typename Coord, typename Real>
	__global__ void CalculateGrowingPointSize(
		DArray<Coord> GrowingPoints,
		DArray<Coord> SeedPoints,
		DArray<Coord> Normals,
		DArray<Coord> outPointNormals,
		DArray<int> SeedOfTriangleId,
		DArray<int> PointOfTriangleId,
		Real thickness,
		Real dx,
		bool direction,
		int pointsLayer
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= SeedPoints.size()) return;

		if (direction == true)
		{
			SeedPoints[tId] -= Normals[tId] * dx * 0.5;
			for (int i = 0; i < pointsLayer; i++)
			{
				GrowingPoints[pointsLayer * tId + i] = SeedPoints[tId] - (Real)(i)*dx * Normals[tId];
				outPointNormals[pointsLayer * tId + i] = Normals[tId];
				PointOfTriangleId[pointsLayer * tId + i] = SeedOfTriangleId[tId];
			}
		}
		else
		{
			SeedPoints[tId] += Normals[tId] * dx * 0.5;
			for (int i = 0; i < pointsLayer; i++)
			{
				GrowingPoints[pointsLayer * tId + i] = SeedPoints[tId] + (Real)(i)*dx * Normals[tId];
				outPointNormals[pointsLayer * tId + i] = -Normals[tId];
				PointOfTriangleId[pointsLayer * tId + i] = SeedOfTriangleId[tId];
			}
		}
	}



	template<typename Coord, typename Real>
	__global__ void CalculateRemovingPointFlag(
		DArray<bool> flags,
		DArray<Coord> positions,
		DArrayList<int> neighbors,
		Real threshold_r
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= flags.size()) return;

		flags[pId] = false;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			if (pId == j) continue;
			Real r = (positions[pId] - positions[j]).norm();
			if ((threshold_r > r) && (pId < j)) //
			{
				flags[pId] = true;
			}
		}
	}


	__global__ void CalculateGhostCounter(
		DArray<int> counter,
		DArray<bool> flags
	)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= flags.size()) return;

		counter[tId] = flags[tId] ? 0 : 1;
	}


	template<typename Coord>
	__global__ void CalculateLastGhostPoints(
		DArray<Coord> lastPoints,
		DArray<Coord> originalPoints,
		DArray<Coord> lastNormals,
		DArray<Coord> originalNormals,
		DArray<int> radix
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= originalPoints.size()) return;

		if (pId == originalPoints.size() - 1 || radix[pId] != radix[pId + 1])
		{
			lastPoints[radix[pId]] = originalPoints[pId];
			lastNormals[radix[pId]] = originalNormals[pId];
		}
	}

	template< typename Coord>
	__global__ void PtsBehindMesh_UpdateTriangleNormal(
		DArray<TopologyModule::Triangle> d_triangles,
		DArray<Coord> d_points,
		DArray<Coord> normal,
		float length,
		bool normalization)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= d_triangles.size()) return;

		int a = d_triangles[pId][0];
		int b = d_triangles[pId][1];
		int c = d_triangles[pId][2];

		float x = (d_points[a][0] + d_points[b][0] + d_points[c][0]) / 3;
		float y = (d_points[a][1] + d_points[b][1] + d_points[c][1]) / 3;
		float z = (d_points[a][2] + d_points[b][2] + d_points[c][2]) / 3;

		Coord ca = d_points[b] - d_points[a];
		Coord cb = d_points[b] - d_points[c];
		Coord dirNormal;

		if (normalization)
			dirNormal = ca.cross(cb).normalize() * -1 * length;
		else
			dirNormal = ca.cross(cb) * -1 * length;

		normal[pId] = dirNormal.normalize();

	}

	template<typename TDataType>
	void PointsBehindMesh<TDataType>::resetStates()
	{
		//Normal<TDataType>::resetStates();

		int triangle_num = this->inTriangleSet()->getData().getTriangles().size();
		//auto& m_triangle_normal = this->stateNormal()->getData();

		if (triangle_num == 0) return;

		this->outPointGrowthDirection()->setValue(this->varGeneratingDirection()->getValue());

		Real dx = this->varSamplingDistance()->getValue();

		if (mTriangleNormal.size() != triangle_num)
		{
			mTriangleNormal.resize(triangle_num);
		}

		if (msquare_1.size() != triangle_num)
		{
			msquare_1.resize(triangle_num);
			msquare_2.resize(triangle_num);
			msquare_3.resize(triangle_num);
			msquare_4.resize(triangle_num);
		}

		if (mBasicVector_x.size() != triangle_num)
		{
			mBasicVector_x.resize(triangle_num);
			mBasicVector_y.resize(triangle_num);
			mBasicVector_z.resize(triangle_num);
			mThinPointSize.resize(triangle_num);
		}

		if (mTriangleTempt.size() != 2 * triangle_num)
			mTriangleTempt.resize(2 * triangle_num);
		if (mVerticesTempt.size() != 4 * triangle_num)
			mVerticesTempt.resize(4 * triangle_num);

		cuExecute(triangle_num,
			PtsBehindMesh_UpdateTriangleNormal,
			this->inTriangleSet()->getData().getTriangles(), //DArray<TopologyModule::Triangle> d_triangles,
			this->inTriangleSet()->getData().getPoints(),
			//DArray<TopologyModule::Edge> edges,
			//DArray<Coord> normal_points,
			mTriangleNormal,
			//DArray<Coord> triangleCenter,
			0.2f,
			true
		);



		cuExecute(triangle_num,
			CalculateSquareFromTriangle,
			msquare_1,
			msquare_2,
			msquare_3,
			msquare_4,
			this->inTriangleSet()->getData().getTriangles(),
			this->inTriangleSet()->getData().getPoints()
		);

		cuExecute(triangle_num,
			CalculateNewTriangle,
			mTriangleTempt,
			msquare_1,
			msquare_2,
			msquare_3,
			msquare_4,
			mVerticesTempt
		);

		auto& temp_triangleSet = this->statePlane()->getData();
		temp_triangleSet.setTriangles(mTriangleTempt);
		temp_triangleSet.setPoints(mVerticesTempt);

		cuExecute(triangle_num,
			CalculateTriangleBasicVector,
			this->inTriangleSet()->getData().getTriangles(),
			this->inTriangleSet()->getData().getPoints(),
			mBasicVector_x,
			mBasicVector_y,
			mBasicVector_z,
			msquare_1,
			msquare_2,
			msquare_3,
			msquare_4,
			this->varSamplingDistance()->getValue()
		);

		cuExecute(triangle_num,
			CalculatePointSizeInSquare,
			mThinPointSize,
			this->inTriangleSet()->getData().getTriangles(),
			this->inTriangleSet()->getData().getPoints(),
			mBasicVector_x,
			mBasicVector_y,
			mBasicVector_z,
			msquare_1,
			msquare_2,
			msquare_3,
			msquare_4,
			this->varSamplingDistance()->getValue()
		);

		Reduction<int> reduce;
		int total_ghostpoint_number = reduce.accumulate(mThinPointSize.begin(), mThinPointSize.size());

		if (total_ghostpoint_number <= 0)
		{
			return;
		}

		DArray<int> tIndex;
		tIndex.assign(mThinPointSize);


		DArray<Coord> mThinPointNormal;
		mThinPointNormal.resize(total_ghostpoint_number);

		mThinPoints.resize(total_ghostpoint_number);
		mSeedOfTriangleId.resize(total_ghostpoint_number);

		thrust::exclusive_scan(thrust::device, tIndex.begin(), tIndex.begin() + tIndex.size(), tIndex.begin());

		cuExecute(triangle_num,
			CalculatePointInSquare,
			mThinPoints,
			mThinPointSize,
			tIndex,
			mThinPointNormal,
			mSeedOfTriangleId,
			this->inTriangleSet()->getData().getTriangles(),
			this->inTriangleSet()->getData().getPoints(),
			mBasicVector_x,
			mBasicVector_y,
			mTriangleNormal, //this->stateNormal()->getData(),
			msquare_1,
			msquare_2,
			msquare_3,
			msquare_4,
			this->varSamplingDistance()->getValue()
		);

		int pointLayerCount = (int)(this->varThickness()->getValue() / this->varSamplingDistance()->getValue());
		pointLayerCount = pointLayerCount < 1 ? 1 : pointLayerCount;

		mThickPoints.resize(mThinPoints.size() * pointLayerCount);

		mPointOfTriangleId.resize(mThinPoints.size() * pointLayerCount);


		DArray<Coord> mThickPointNormals;
		mThickPointNormals.resize(mThinPoints.size() * pointLayerCount);

		cuExecute(mThinPoints.size(),
			CalculateGrowingPointSize,
			mThickPoints,
			mThinPoints,
			mThinPointNormal,
			mThickPointNormals,
			mSeedOfTriangleId,
			mPointOfTriangleId,
			this->varThickness()->getValue(),
			dx,
			this->varGeneratingDirection()->getValue(),
			pointLayerCount
		);

		if (mRemovingFlag.size() != mThickPoints.size())
		{
			mRemovingFlag.resize(mThickPoints.size());
		}

		this->statePosition()->getData().assign(mThickPoints);
		this->statePosition()->connect(m_NeighborPointQuery->inPosition());
		m_NeighborPointQuery->inRadius()->setValue(this->varSamplingDistance()->getValue() * 1.0);
		m_NeighborPointQuery->update();

		cuExecute(mThickPoints.size(),
			CalculateRemovingPointFlag,
			mRemovingFlag,
			mThickPoints,
			m_NeighborPointQuery->outNeighborIds()->getData(),
			0.75f * this->varSamplingDistance()->getValue()
		);

		DArray<int> ghost_counter(mRemovingFlag.size());

		cuExecute(mThickPoints.size(),
			CalculateGhostCounter,
			ghost_counter,
			mRemovingFlag
		);

		int last_ghostpoint_number = reduce.accumulate(ghost_counter.begin(), ghost_counter.size());
		scan.exclusive(ghost_counter.begin(), ghost_counter.size());

		DArray<Coord> lastPoints(last_ghostpoint_number);
		auto& lastPointNormal = this->statePointNormal()->getData();
		lastPointNormal.resize(last_ghostpoint_number);

		this->statePointNormal()->resize(last_ghostpoint_number);

		if (last_ghostpoint_number > 0)
		{
			cuExecute(mThickPoints.size(),
				CalculateLastGhostPoints,
				lastPoints,
				mThickPoints,
				this->statePointNormal()->getData(),
				mThickPointNormals,
				ghost_counter
			);
		}

		auto& ghostPointSet = this->statePointSet()->getData();
		this->statePosition()->getData().clear();
		this->statePosition()->getData().assign(lastPoints);
		this->statePointSet()->getData().clear();
		this->statePointSet()->getData().setPoints(lastPoints);

		this->outSamplingDistance()->setValue(this->varSamplingDistance()->getValue());
		this->statePointBelongTriangleIndex()->assign(mPointOfTriangleId);




		tIndex.clear();
		lastPoints.clear();
		ghost_counter.clear();
		mThinPointNormal.clear();
		mThickPointNormals.clear();

	}


	DEFINE_CLASS(PointsBehindMesh);
}