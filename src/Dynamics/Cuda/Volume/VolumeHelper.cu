#include "VolumeHelper.h"

#include <thrust/sort.h>

namespace dyno
{
	template<typename TCoord>
	struct NodeCmp
	{
		DYN_FUNC bool operator()(const VoxelOctreeNode<TCoord>& A, const VoxelOctreeNode<TCoord>& B)
		{
			return A > B;
		}
	};

	template <typename Real, typename Coord>
	GPU_FUNC int SO_ComputeGrid(
		int& nx_lo,
		int& ny_lo,
		int& nz_lo,
		int& nx_hi,
		int& ny_hi,
		int& nz_hi,
		Coord surf_p,
		Coord surf_q,
		Coord surf_r,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_,
		int extend_band)
	{
		Coord fp = (surf_p - origin_) / dx_;
		Coord fq = (surf_q - origin_) / dx_;
		Coord fr = (surf_r - origin_) / dx_;

		nx_hi = clamp(int(maximum(fp[0], maximum(fq[0], fr[0]))) + extend_band, 0, nx_ - 1);
		ny_hi = clamp(int(maximum(fp[1], maximum(fq[1], fr[1]))) + extend_band, 0, ny_ - 1);
		nz_hi = clamp(int(maximum(fp[2], maximum(fq[2], fr[2]))) + extend_band, 0, nz_ - 1);

		nx_lo = clamp(int(minimum(fp[0], minimum(fq[0], fr[0]))) - extend_band, 0, nx_ - 1);
		ny_lo = clamp(int(minimum(fp[1], minimum(fq[1], fr[1]))) - extend_band, 0, ny_ - 1);
		nz_lo = clamp(int(minimum(fp[2], minimum(fq[2], fr[2]))) - extend_band, 0, nz_ - 1);

		if ((nx_hi % 2) != 1) nx_hi++;
		if ((ny_hi % 2) != 1) ny_hi++;
		if ((nz_hi % 2) != 1) nz_hi++;
		if ((nx_lo % 2) != 0) nx_lo--;
		if ((ny_lo % 2) != 0) ny_lo--;
		if ((nz_lo % 2) != 0) nz_lo--;

		return (nz_hi - nz_lo + 1) * (ny_hi - ny_lo + 1) * (nx_hi - nx_lo + 1);
	}

	template <typename Real, typename Coord, typename Triangle>
	__global__ void SO_SurfaceCount(
		DArray<int> counter,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_,
		int extend_band)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		int p = surf_triangles[tId][0];
		int q = surf_triangles[tId][1];
		int r = surf_triangles[tId][2];

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		counter[tId] = SO_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, surf_points[p], surf_points[q], surf_points[r], origin_, dx_, nx_, ny_, nz_, extend_band);
	}

	template <typename Real, typename Coord, typename Triangle>
	__global__ void SO_SurfaceInit(
		DArray<PositionNode> nodes,
		DArray<int> counter,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_,
		int extend_band)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		int p = surf_triangles[tId][0];
		int q = surf_triangles[tId][1];
		int r = surf_triangles[tId][2];

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		int num = SO_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, surf_points[p], surf_points[q], surf_points[r], origin_, dx_, nx_, ny_, nz_, extend_band);

		if (num > 0)
		{
			int acc_num = 0;
			for (int k = nz_lo; k <= nz_hi; k++) {
				for (int j = ny_lo; j <= ny_hi; j++) {
					for (int i = nx_lo; i <= nx_hi; i++)
					{
						OcKey index = CalculateMortonCode(i, j, k);
						nodes[counter[tId] + acc_num] = PositionNode(tId, index);

						acc_num++;
					}
				}
			}
		}
	}

	__global__ void SO_CountNonRepeatedPosition(
		DArray<int> counter,
		DArray<PositionNode> nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if ((tId == 0 || nodes[tId].position_index != nodes[tId - 1].position_index))
		{
			counter[tId] = 1;
		}
	}

	template <typename Coord, typename Triangle, typename Tri2Edg, typename Edge>
	GPU_FUNC void SO_ComputeObjectAndNormal(
		Coord& pobject,
		Coord& pnormal,
		DArray<Tri2Edg>& t2e,
		DArray<Edge>& edge,
		DArray<Coord>& edgeN,
		DArray<Coord>& vertexN,
		DArray<Triangle>& surf_triangles,
		DArray<Coord>& surf_points,
		Coord ppos,
		int surf_id)
	{
		int p = surf_triangles[surf_id][0];
		int q = surf_triangles[surf_id][1];
		int r = surf_triangles[surf_id][2];
		Coord p0 = surf_points[p];
		Coord p1 = surf_points[q];
		Coord p2 = surf_points[r];

		int eid0 = t2e[surf_id][0];
		int eid1 = t2e[surf_id][1];
		int eid2 = t2e[surf_id][2];

		Coord dir = p0 - ppos;
		Coord e0 = p1 - p0;
		Coord e1 = p2 - p0;
		Coord e2 = p2 - p1;
		Real a = e0.dot(e0);
		Real b = e0.dot(e1);
		Real c = e1.dot(e1);
		Real d = e0.dot(dir);
		Real e = e1.dot(dir);
		Real f = dir.dot(dir);

		Real det = a * c - b * b;
		Real s = b * e - c * d;
		Real t = b * d - a * e;

		Real maxL = maximum(maximum(e0.norm(), e1.norm()), e2.norm());
		//handle degenerate triangles
		if (det < REAL_EPSILON * maxL * maxL)
		{
			Real g = e2.normSquared();
			Real l_max = a;

			Coord op0 = p0;
			Coord op1 = p1;
			EKey oe(p, q);
			if (c > l_max)
			{
				op0 = p0;
				op1 = p2;
				oe = EKey(p, r);

				l_max = c;
			}
			if (g > l_max)
			{
				op0 = p1;
				op1 = p2;
				oe = EKey(q, r);
			}

			Coord el = ppos - op0;
			Coord edir = op1 - op0;
			if (edir.normSquared() < REAL_EPSILON_SQUARED)
			{
				pobject = surf_points[oe[0]];
				pnormal = vertexN[oe[0]];
				return;
			}

			Real et = el.dot(edir) / edir.normSquared();

			if (et <= 0)
			{
				pobject = surf_points[oe[0]];
				pnormal = vertexN[oe[0]];
				return;
			}
			else if (et >= 1)
			{
				pobject = surf_points[oe[1]];
				pnormal = vertexN[oe[1]];
				return;
			}
			else
			{
				Coord eq = op0 + et * edir;
				pobject = eq;
				if (oe == EKey(edge[eid0][0], edge[eid0][1]))
				{
					pnormal = edgeN[eid0];
					return;
				}
				else if (oe == EKey(edge[eid1][0], edge[eid1][1]))
				{
					pnormal = edgeN[eid1];
					return;
				}
				else if (oe == EKey(edge[eid2][0], edge[eid2][1]))
				{
					pnormal = edgeN[eid2];
					return;
				}
			}
		}
		if (s + t <= det)
		{
			if (s < 0)
			{
				if (t < 0)
				{
					//region 4
					s = 0;
					t = 0;
				}
				else
				{
					// region 3
					s = 0;
					t = (e >= 0 ? 0 : (-e >= c ? 1 : -e / c));
				}
			}
			else
			{
				if (t < 0)
				{
					//region 5
					s = (d >= 0 ? 0 : (-d >= a ? 1 : -d / a));
					t = 0;
				}
				else
				{
					//region 0
					Real invDet = 1 / det;
					s *= invDet;
					t *= invDet;
				}
			}
		}
		else
		{
			if (s < 0)
			{
				//region 2
				s = 0;
				t = 1;
			}
			else if (t < 0)
			{
				//region 6
				s = 1;
				t = 0;
			}
			else
			{
				//region 1
				Real numer = c + e - b - d;
				if (numer <= 0) {
					s = 0;
				}
				else {
					Real denom = a - 2 * b + c; // positive quantity
					s = (numer >= denom ? 1 : numer / denom);
				}
				t = 1 - s;
			}
		}
		pobject = (p0 + s * e0 + t * e1);
		if (s == 0 && t == 0)
		{
			pnormal = vertexN[p];
			return;
		}
		else if (s == 0 && t == 1)
		{
			pnormal = vertexN[r];
			return;
		}
		else if (s == 1 && t == 0)
		{
			pnormal = vertexN[q];
			return;
		}
		else if (s == 0 && t < 1)
		{
			pnormal = edgeN[eid2];
			return;
		}
		else if (s < 1 && t == 0)
		{
			pnormal = edgeN[eid0];
			return;
		}
		else if (s + t == 1)
		{
			pnormal = edgeN[eid1];
			return;
		}
		else
		{
			pnormal = (p1 - p0).cross(p2 - p0);
			pnormal.normalize();
			return;
		}
	}

	//注意counter中第i+1个元素存的是0-i元素的和
	template <typename Real, typename Coord, typename Triangle, typename Tri2Edg, typename Edge>
	__global__ void SO_FetchNonRepeatedPosition(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<PositionNode> all_nodes,
		DArray<int> counter,
		DArray<Tri2Edg> t2e,
		DArray<Edge> edge,
		DArray<Coord> edgeN,
		DArray<Coord> vertexN,
		DArray<Triangle> surf_triangles,
		DArray<Coord> surf_points,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if ((tId == 0 || all_nodes[tId].position_index != all_nodes[tId - 1].position_index))
		{
			Coord pnormal(0), pobject(0);
			int surf_g = all_nodes[tId].surface_index;

			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(all_nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			SO_ComputeObjectAndNormal(pobject, pnormal, t2e, edge, edgeN, vertexN, surf_triangles, surf_points, pos, surf_g);

			Real sign = (pos - pobject).dot(pnormal) < Real(0) ? Real(-1) : Real(1);
			Real dist = sign * (pos - pobject).norm();

			int acc_num = 1;
			while (((tId + acc_num) < counter.size()) && (all_nodes[tId + acc_num].position_index == all_nodes[tId].position_index))
			{
				int surf_g_i = all_nodes[tId + acc_num].surface_index;

				Coord pnormal_i(0), pobject_i(0);
				SO_ComputeObjectAndNormal(pobject_i, pnormal_i, t2e, edge, edgeN, vertexN, surf_triangles, surf_points, pos, surf_g_i);

				Real sign_i = (pos - pobject_i).dot(pnormal_i) < Real(0) ? Real(-1) : Real(1);
				Real dist_i = sign_i * (pos - pobject_i).norm();

				if (std::abs(dist_i) < std::abs(dist))
				{
					pnormal = pnormal_i;
					pobject = pobject_i;
					dist = dist_i;
				}
				acc_num++;
			}

			VoxelOctreeNode<Coord> mc((Level)0, gnx, gny, gnz, pos);
			mc.setValueLocation(counter[tId]);

			nodes[counter[tId]] = mc;
			nodes_value[counter[tId]] = dist;
			nodes_object[counter[tId]] = pobject;
			nodes_normal[counter[tId]] = pnormal;

			x_index[counter[tId]] = IndexNode((gnz * nx_ * ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx * ny_ * nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny * nz_ * nx_ + gnx * nz_ + gnz), counter[tId]);
		}
	}

	template <typename Coord>
	__global__ void SO_UpdateBottomNeighbors(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		int xnx = (x_index[tId].xyz_index) % nx_;
		if ((xnx != 0 && tId > 0) && (x_index[tId - 1].xyz_index == (x_index[tId].xyz_index - 1)))
			nodes[x_index[tId].node_index].m_neighbor[0] = x_index[tId - 1].node_index;
		if ((xnx != (nx_ - 1) && tId < (nodes.size() - 1)) && (x_index[tId + 1].xyz_index == (x_index[tId].xyz_index + 1)))
			nodes[x_index[tId].node_index].m_neighbor[1] = x_index[tId + 1].node_index;

		int yny = (y_index[tId].xyz_index) % ny_;
		if ((yny != 0 && tId > 0) && (y_index[tId - 1].xyz_index == (y_index[tId].xyz_index - 1)))
			nodes[y_index[tId].node_index].m_neighbor[2] = y_index[tId - 1].node_index;
		if ((yny != (ny_ - 1) && tId < (nodes.size() - 1)) && (y_index[tId + 1].xyz_index == (y_index[tId].xyz_index + 1)))
			nodes[y_index[tId].node_index].m_neighbor[3] = y_index[tId + 1].node_index;

		int znz = (z_index[tId].xyz_index) % nz_;
		if ((znz != 0 && tId > 0) && (z_index[tId - 1].xyz_index == (z_index[tId].xyz_index - 1)))
			nodes[z_index[tId].node_index].m_neighbor[4] = z_index[tId - 1].node_index;
		if ((znz != (nz_ - 1) && tId < (nodes.size() - 1)) && (z_index[tId + 1].xyz_index == (z_index[tId].xyz_index + 1)))
			nodes[z_index[tId].node_index].m_neighbor[5] = z_index[tId + 1].node_index;
	}

	template <typename Real, typename Coord>
	__global__ void SO_ComputeUpLevelGrids(
		DArray<VoxelOctreeNode<Coord>> up_level_nodes,
		DArray<Real> up_level_value,
		DArray<Coord> up_level_object,
		DArray<Coord> up_level_normal,
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		Coord origin_,
		Real dx_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		Level g_level = nodes[tId].level();
		g_level++;

		OcIndex fnx, fny, fnz;
		OcKey cg_key = nodes[tId].key();
		OcKey fg_key = cg_key >> 6;
		RecoverFromMortonCode(fg_key, fnx, fny, fnz);

		OcIndex gnx, gny, gnz;
		int gtId = 7 - (tId % 8);
		RecoverFromMortonCode((OcKey)gtId, gnx, gny, gnz);
		Real dx_l = Real(pow(Real(2), int(g_level)) * dx_);
		Coord pos(origin_[0] + (fnx * 2 + gnx + 0.5) * dx_l, origin_[1] + (fny * 2 + gny + 0.5) * dx_l, origin_[2] + (fnz * 2 + gnz + 0.5) * dx_l);

		int ctId = tId - (tId % 8);

		Real sign = (pos - nodes_object[ctId]).dot(nodes_normal[ctId]) < Real(0) ? Real(-1) : Real(1);
		Real node_value = sign * (pos - nodes_object[ctId]).norm();
		int node_index = 0;
		for (int i = 1; i < 8; i++)
		{
			Real sign_i = (pos - nodes_object[ctId + i]).dot(nodes_normal[ctId + i]) < Real(0) ? Real(-1) : Real(1);
			Real node_value_i = sign_i * (pos - nodes_object[ctId + i]).norm();

			if (abs(node_value_i) < abs(node_value))
			{
				node_value = node_value_i;
				node_index = i;
			}
		}

		up_level_nodes[tId] = VoxelOctreeNode<Coord>(g_level, (2 * fnx + gnx), (2 * fny + gny), (2 * fnz + gnz), pos);
		up_level_nodes[tId].setValueLocation(tId);
		up_level_value[tId] = node_value;
		up_level_object[tId] = nodes_object[ctId + node_index];
		up_level_normal[tId] = nodes_normal[ctId + node_index];

		int gIndex = (cg_key >> 3) & 7U;
		if (gIndex == gtId)
		{
			up_level_nodes[tId].setMidsideNode();
			up_level_nodes[tId].setChildIndex(ctId);
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_CountNonRepeatedGrids(
		DArray<int> counter,
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if ((tId == 0 || nodes[tId].key() != nodes[tId - 1].key()))
		{
			int child_id;
			bool ismidside = nodes[tId].midside();
			if (ismidside) child_id = nodes[tId].child();

			int counter_index = tId;
			Real counter_dist = std::abs(nodes_value[nodes[tId].value()]);
			int rep_num = 1;
			while (((tId + rep_num) < counter.size()) && (nodes[tId].key() == nodes[tId + rep_num].key()))
			{

				ismidside = ismidside || (nodes[tId + rep_num].midside());
				if (nodes[tId + rep_num].midside())
					child_id = nodes[tId + rep_num].child();

				if (abs(nodes_value[nodes[tId + rep_num].value()]) < counter_dist)
				{
					counter_index = tId + rep_num;
					counter_dist = abs(nodes_value[nodes[tId + rep_num].value()]);
				}
				rep_num++;
			}
			if (ismidside)
			{
				nodes[counter_index].setMidsideNode();
				nodes[counter_index].setChildIndex(child_id);
			}
			counter[counter_index] = 1;
		}
	}

	//注意counter中第i+1个元素存的是0-i元素的和
	template <typename Real, typename Coord>
	__global__ void SO_FetchNonRepeatedGrids(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<VoxelOctreeNode<Coord>> all_nodes,
		DArray<Real> all_nodes_value,
		DArray<Coord> all_nodes_object,
		DArray<Coord> all_nodes_normal,
		DArray<int> counter,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId < (counter.size() - 1) && counter[tId] != counter[tId + 1])
		{
			nodes[counter[tId]] = all_nodes[tId];
			nodes[counter[tId]].setValueLocation(counter[tId]);
			nodes_value[counter[tId]] = all_nodes_value[all_nodes[tId].value()];
			nodes_object[counter[tId]] = all_nodes_object[all_nodes[tId].value()];
			nodes_normal[counter[tId]] = all_nodes_normal[all_nodes[tId].value()];

			OcIndex gnx, gny, gnz;
			OcKey g_key = all_nodes[tId].key();
			RecoverFromMortonCode(g_key, gnx, gny, gnz);
			x_index[counter[tId]] = IndexNode((gnz * nx_ * ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx * ny_ * nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny * nz_ * nx_ + gnx * nz_ + gnz), counter[tId]);
		}
		else if (tId == (counter.size() - 1) && (counter[tId] < nodes.size()))
		{
			nodes[counter[tId]] = all_nodes[tId];
			nodes[counter[tId]].setValueLocation(counter[tId]);
			nodes_value[counter[tId]] = all_nodes_value[all_nodes[tId].value()];
			nodes_object[counter[tId]] = all_nodes_object[all_nodes[tId].value()];
			nodes_normal[counter[tId]] = all_nodes_normal[all_nodes[tId].value()];

			OcIndex gnx, gny, gnz;
			OcKey g_key = all_nodes[tId].key();
			RecoverFromMortonCode(g_key, gnx, gny, gnz);
			x_index[counter[tId]] = IndexNode((gnz * nx_ * ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx * ny_ * nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny * nz_ * nx_ + gnx * nz_ + gnz), counter[tId]);
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_FIMUpLevelGrids(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		int xnx = (x_index[tId].xyz_index) % nx_;
		if ((xnx != 0 && tId > 0) && (x_index[tId - 1].xyz_index == (x_index[tId].xyz_index - 1)))
			nodes[x_index[tId].node_index].m_neighbor[0] = x_index[tId - 1].node_index;
		if ((xnx != (nx_ - 1) && tId < (nodes.size() - 1)) && (x_index[tId + 1].xyz_index == (x_index[tId].xyz_index + 1)))
			nodes[x_index[tId].node_index].m_neighbor[1] = x_index[tId + 1].node_index;

		int yny = (y_index[tId].xyz_index) % ny_;
		if ((yny != 0 && tId > 0) && (y_index[tId - 1].xyz_index == (y_index[tId].xyz_index - 1)))
			nodes[y_index[tId].node_index].m_neighbor[2] = y_index[tId - 1].node_index;
		if ((yny != (ny_ - 1) && tId < (nodes.size() - 1)) && (y_index[tId + 1].xyz_index == (y_index[tId].xyz_index + 1)))
			nodes[y_index[tId].node_index].m_neighbor[3] = y_index[tId + 1].node_index;

		int znz = (z_index[tId].xyz_index) % nz_;
		if ((znz != 0 && tId > 0) && (z_index[tId - 1].xyz_index == (z_index[tId].xyz_index - 1)))
			nodes[z_index[tId].node_index].m_neighbor[4] = z_index[tId - 1].node_index;
		if ((znz != (nz_ - 1) && tId < (nodes.size() - 1)) && (z_index[tId + 1].xyz_index == (z_index[tId].xyz_index + 1)))
			nodes[z_index[tId].node_index].m_neighbor[5] = z_index[tId + 1].node_index;

		__syncthreads();

		for (int i = 0; i < 6; i++)
		{
			if (nodes[tId].m_neighbor[i] != EMPTY)
			{
				int nb_id = nodes[tId].m_neighbor[i];
				Real sign = (nodes[tId].position() - nodes_object[nb_id]).dot(nodes_normal[nb_id]) < Real(0) ? Real(-1) : Real(1);
				Real dist = sign * (nodes[tId].position() - nodes_object[nb_id]).norm();

				if (abs(dist) < abs(nodes_value[tId]))
				{
					nodes_value[tId] = dist;
					nodes_object[tId] = nodes_object[nb_id];
					nodes_normal[tId] = nodes_normal[nb_id];
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_ComputeTopLevelGrids(
		DArray<VoxelOctreeNode<Coord>> top_level_nodes,
		DArray<Coord> top_level_object,
		DArray<Coord> top_level_normal,
		DArray<int> top_count,
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		Coord origin_,
		Real dx_,
		int tnx,
		int tny,
		int tnz,
		Level grid_level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		Level cg_level = nodes[tId].level();
		if (cg_level != (grid_level - 1)) return;

		OcIndex gnx, gny, gnz;
		OcKey cg_key = nodes[tId].key();
		OcKey g_key = cg_key >> 3;
		RecoverFromMortonCode(g_key, gnx, gny, gnz);

		int index = gnx + gny * tnx + gnz * tnx * tny;

		Coord gpos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

		auto mc = VoxelOctreeNode<Coord>(grid_level, g_key);

		if ((tId % 8) == 0)
		{
			top_level_nodes[index] = mc;
			top_level_nodes[index].setMidsideNode();
			top_level_nodes[index].setChildIndex(tId);
			top_level_nodes[index].setPosition(gpos);
		}

		int gIndex = cg_key & 7U;
		top_level_object[8 * index + gIndex] = nodes_object[tId];
		top_level_normal[8 * index + gIndex] = nodes_normal[tId];
		top_count[index] = 1;
	}

	template <typename Real, typename Coord>
	__global__ void SO_UpdateTopLevelGrids(
		DArray<VoxelOctreeNode<Coord>> top_level_nodes,
		DArray<Real> top_level_val,
		DArray<Coord> top_level_object,
		DArray<Coord> top_level_normal,
		DArray<int> node_ind,
		DArray<Coord> top_object_buf,
		DArray<Coord> top_normal_buf,
		Coord origin_,
		Real dx_,
		int tnx,
		int tny,
		int tnz,
		Level grid_level)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= top_level_nodes.size()) return;

		int gnz = (int)(tId / (tnx * tny));
		int gny = (int)((tId % (tnx * tny)) / tnx);
		int gnx = (tId % (tnx * tny)) % tnx;

		if (node_ind[tId] == 1)
		{
			int index = 0;
			Real sign = (top_level_nodes[tId].position() - top_object_buf[8 * tId]).dot(top_normal_buf[8 * tId]) < Real(0) ? Real(-1) : Real(1);
			Real min_value = sign * (top_level_nodes[tId].position() - top_object_buf[8 * tId]).norm();

			for (int i = 1; i <= 7; i++)
			{
				Real sign_i = (top_level_nodes[tId].position() - top_object_buf[8 * tId + i]).dot(top_normal_buf[8 * tId + i]) < Real(0) ? Real(-1) : Real(1);
				Real min_value_i = sign_i * (top_level_nodes[tId].position() - top_object_buf[8 * tId + i]).norm();

				if (abs(min_value) > abs(min_value_i))
				{
					index = i;
					min_value = min_value_i;
				}
			}
			top_level_val[tId] = min_value;
			top_level_object[tId] = top_object_buf[8 * tId + index];
			top_level_normal[tId] = top_normal_buf[8 * tId + index];
			top_level_nodes[tId].setValueLocation(tId);
		}
		else
		{
			Coord gpos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			auto mc = VoxelOctreeNode<Coord>(grid_level, gnx, gny, gnz);
			mc.setPosition(gpos);
			mc.setValueLocation(tId);
			top_level_nodes[tId] = mc;
		}

		//update m_neighbor
		if (gnx > 0)
			top_level_nodes[tId].m_neighbor[0] = tId - 1;
		if (gnx < (tnx - 1))
			top_level_nodes[tId].m_neighbor[1] = tId + 1;
		if (gny > 0)
			top_level_nodes[tId].m_neighbor[2] = tId - tnx;
		if (gny < (tny - 1))
			top_level_nodes[tId].m_neighbor[3] = tId + tnx;
		if (gnz > 0)
			top_level_nodes[tId].m_neighbor[4] = tId - tnx * tny;
		if (gnz < (tnz - 1))
			top_level_nodes[tId].m_neighbor[5] = tId + tnx * tny;
	}

	template <typename Real, typename Coord>
	DYN_FUNC void SO_ComputeGridWithNeighbor(
		bool& update,
		int& update_id,
		Real& value_id,
		int index_id,
		DArray<Coord>& nodes_object,
		DArray<Coord>& nodes_normal,
		Coord grid_pos)
	{
		Real sign = (grid_pos - nodes_object[index_id]).dot(nodes_normal[index_id]) < Real(0) ? Real(-1) : Real(1);
		Real dist_value = sign * (grid_pos - nodes_object[index_id]).norm();

		if (abs(dist_value) < abs(value_id))
		{
			update_id = index_id;
			value_id = dist_value;
			update = true;
		}
	}


	template <typename Real, typename Coord>
	__global__ void SO_FIMComputeTopLevelGrids(
		DArray<VoxelOctreeNode<Coord>> top_level_nodes,
		DArray<Real> top_level_value,
		DArray<Coord> top_level_object,
		DArray<Coord> top_level_normal,
		DArray<int> node_ind,
		DArray<Coord> object_temp,
		DArray<Coord> normal_temp,
		DArray<int> node_ind_temp,
		int tnx,
		int tny,
		int tnz)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= top_level_nodes.size()) return;

		int gnz = (int)(tId / (tnx * tny));
		int gny = (int)((tId % (tnx * tny)) / tnx);
		int gnx = (tId % (tnx * tny)) % tnx;

		Coord gpos = top_level_nodes[tId].position();

		bool update = false;
		int update_id;
		Real value = std::numeric_limits<Real>::max();
		if (node_ind_temp[tId] == 1)
		{
			value = top_level_value[tId];
			update_id = tId;
		}
		if (gnx > 0)
			if (node_ind_temp[tId - 1] == 1)
			{
				SO_ComputeGridWithNeighbor(update, update_id, value, (tId - 1), object_temp, normal_temp, gpos);
			}
		if (gnx < (tnx - 1))
			if (node_ind_temp[tId + 1] == 1)
			{
				SO_ComputeGridWithNeighbor(update, update_id, value, (tId + 1), object_temp, normal_temp, gpos);
			}
		if (gny > 0)
			if (node_ind_temp[tId - tnx] == 1)
			{
				SO_ComputeGridWithNeighbor(update, update_id, value, (tId - tnx), object_temp, normal_temp, gpos);
			}
		if (gny < (tny - 1))
			if (node_ind_temp[tId + tnx] == 1)
			{
				SO_ComputeGridWithNeighbor(update, update_id, value, (tId + tnx), object_temp, normal_temp, gpos);
			}
		if (gnz > 0)
			if (node_ind_temp[tId - tnx * tny] == 1)
			{
				SO_ComputeGridWithNeighbor(update, update_id, value, (tId - tnx * tny), object_temp, normal_temp, gpos);
			}
		if (gnz < (tnz - 1))
			if (node_ind_temp[tId + tnx * tny] == 1)
			{
				SO_ComputeGridWithNeighbor(update, update_id, value, (tId + tnx * tny), object_temp, normal_temp, gpos);
			}

		if (update)
		{
			top_level_value[tId] = value;
			node_ind[tId] = 1;
			top_level_object[tId] = object_temp[update_id];
			top_level_normal[tId] = normal_temp[update_id];
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_CollectionGrids(
		DArray<VoxelOctreeNode<Coord>> total_nodes,
		DArray<Real> total_value,
		DArray<Coord> total_object,
		DArray<Coord> total_normal,
		DArray<VoxelOctreeNode<Coord>> level0_nodes,
		DArray<Real> level0_value,
		DArray<Coord> level0_object,
		DArray<Coord> level0_normal,
		DArray<VoxelOctreeNode<Coord>> level1_nodes,
		DArray<Real> level1_value,
		DArray<Coord> level1_object,
		DArray<Coord> level1_normal,
		int level0_num,
		int level0_child_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= total_nodes.size()) return;

		if (tId >= level0_num)
		{
			total_nodes[tId] = level1_nodes[(tId - level0_num)];
			total_nodes[tId].setValueLocation(tId);
			if (total_nodes[tId].midside() == true)
				total_nodes[tId].plusChildIndex(level0_num - level0_child_num);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level0_num;
			}

			total_value[tId] = level1_value[(tId - level0_num)];
			total_object[tId] = level1_object[(tId - level0_num)];
			total_normal[tId] = level1_normal[(tId - level0_num)];
		}
		else
		{
			total_nodes[tId] = level0_nodes[tId];
			total_value[tId] = level0_value[tId];
			total_object[tId] = level0_object[tId];
			total_normal[tId] = level0_normal[tId];
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_CollectionGridsThree(
		DArray<VoxelOctreeNode<Coord>> total_nodes,
		DArray<Real> total_value,
		DArray<Coord> total_object,
		DArray<Coord> total_normal,
		DArray<VoxelOctreeNode<Coord>> level0_nodes,
		DArray<Real> level0_value,
		DArray<Coord> level0_object,
		DArray<Coord> level0_normal,
		DArray<VoxelOctreeNode<Coord>> level1_nodes,
		DArray<Real> level1_value,
		DArray<Coord> level1_object,
		DArray<Coord> level1_normal,
		DArray<VoxelOctreeNode<Coord>> levelT_nodes,
		DArray<Real> levelT_value,
		DArray<Coord> levelT_object,
		DArray<Coord> levelT_normal,
		int level0_num,
		int level1_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= total_nodes.size()) return;

		if (tId >= level1_num)
		{
			total_nodes[tId] = levelT_nodes[(tId - level1_num)];
			total_nodes[tId].setValueLocation(tId);
			if (total_nodes[tId].midside() == true)
				total_nodes[tId].plusChildIndex(level0_num);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level1_num;
			}

			total_value[tId] = levelT_value[(tId - level1_num)];
			total_object[tId] = levelT_object[(tId - level1_num)];
			total_normal[tId] = levelT_normal[(tId - level1_num)];
		}
		else if (tId >= level0_num)
		{
			total_nodes[tId] = level1_nodes[(tId - level0_num)];
			total_nodes[tId].setValueLocation(tId);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level0_num;
			}

			total_value[tId] = level1_value[(tId - level0_num)];
			total_object[tId] = level1_object[(tId - level0_num)];
			total_normal[tId] = level1_normal[(tId - level0_num)];
		}
		else
		{
			total_nodes[tId] = level0_nodes[tId];
			total_nodes[tId].setValueLocation(tId);

			total_value[tId] = level0_value[tId];
			total_object[tId] = level0_object[tId];
			total_normal[tId] = level0_normal[tId];
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_CollectionGridsFour(
		DArray<VoxelOctreeNode<Coord>> total_nodes,
		DArray<Real> total_value,
		DArray<Coord> total_object,
		DArray<Coord> total_normal,
		DArray<VoxelOctreeNode<Coord>> level0_nodes,
		DArray<Real> level0_value,
		DArray<Coord> level0_object,
		DArray<Coord> level0_normal,
		DArray<VoxelOctreeNode<Coord>> level1_nodes,
		DArray<Real> level1_value,
		DArray<Coord> level1_object,
		DArray<Coord> level1_normal,
		DArray<VoxelOctreeNode<Coord>> level2_nodes,
		DArray<Real> level2_value,
		DArray<Coord> level2_object,
		DArray<Coord> level2_normal,
		DArray<VoxelOctreeNode<Coord>> levelT_nodes,
		DArray<Real> levelT_value,
		DArray<Coord> levelT_object,
		DArray<Coord> levelT_normal,
		int level0_num,
		int level1_num,
		int level2_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= total_nodes.size()) return;

		if (tId >= level2_num)
		{
			total_nodes[tId] = levelT_nodes[(tId - level2_num)];
			total_nodes[tId].setValueLocation(tId);
			if (total_nodes[tId].midside() == true)
				total_nodes[tId].plusChildIndex(level1_num);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level2_num;
			}

			total_value[tId] = levelT_value[(tId - level2_num)];
			total_object[tId] = levelT_object[(tId - level2_num)];
			total_normal[tId] = levelT_normal[(tId - level2_num)];
		}
		else if (tId >= level1_num)
		{
			total_nodes[tId] = level2_nodes[(tId - level1_num)];
			total_nodes[tId].setValueLocation(tId);
			if (total_nodes[tId].midside() == true)
				total_nodes[tId].plusChildIndex(level0_num);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level1_num;
			}

			total_value[tId] = level2_value[(tId - level1_num)];
			total_object[tId] = level2_object[(tId - level1_num)];
			total_normal[tId] = level2_normal[(tId - level1_num)];
		}
		else if (tId >= level0_num)
		{
			total_nodes[tId] = level1_nodes[(tId - level0_num)];
			total_nodes[tId].setValueLocation(tId);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level0_num;
			}

			total_value[tId] = level1_value[(tId - level0_num)];
			total_object[tId] = level1_object[(tId - level0_num)];
			total_normal[tId] = level1_normal[(tId - level0_num)];
		}
		else
		{
			total_nodes[tId] = level0_nodes[tId];
			total_value[tId] = level0_value[tId];
			total_object[tId] = level0_object[tId];
			total_normal[tId] = level0_normal[tId];
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_CollectionGridsFive(
		DArray<VoxelOctreeNode<Coord>> total_nodes,
		DArray<Real> total_value,
		DArray<Coord> total_object,
		DArray<Coord> total_normal,
		DArray<VoxelOctreeNode<Coord>> level0_nodes,
		DArray<Real> level0_value,
		DArray<Coord> level0_object,
		DArray<Coord> level0_normal,
		DArray<VoxelOctreeNode<Coord>> level1_nodes,
		DArray<Real> level1_value,
		DArray<Coord> level1_object,
		DArray<Coord> level1_normal,
		DArray<VoxelOctreeNode<Coord>> level2_nodes,
		DArray<Real> level2_value,
		DArray<Coord> level2_object,
		DArray<Coord> level2_normal,
		DArray<VoxelOctreeNode<Coord>> level3_nodes,
		DArray<Real> level3_value,
		DArray<Coord> level3_object,
		DArray<Coord> level3_normal,
		DArray<VoxelOctreeNode<Coord>> levelT_nodes,
		DArray<Real> levelT_value,
		DArray<Coord> levelT_object,
		DArray<Coord> levelT_normal,
		int level0_num,
		int level1_num,
		int level2_num,
		int level3_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= total_nodes.size()) return;

		if (tId >= level3_num)
		{
			total_nodes[tId] = levelT_nodes[(tId - level3_num)];
			total_value[tId] = levelT_value[(tId - level3_num)];
			total_object[tId] = levelT_object[(tId - level3_num)];
			total_normal[tId] = levelT_normal[(tId - level3_num)];
		}
		else if (tId >= level2_num)
		{
			total_nodes[tId] = level3_nodes[(tId - level2_num)];
			total_value[tId] = level3_value[(tId - level2_num)];
			total_object[tId] = level3_object[(tId - level2_num)];
			total_normal[tId] = level3_normal[(tId - level2_num)];
		}
		else if (tId >= level1_num)
		{
			total_nodes[tId] = level2_nodes[(tId - level1_num)];
			total_value[tId] = level2_value[(tId - level1_num)];
			total_object[tId] = level2_object[(tId - level1_num)];
			total_normal[tId] = level2_normal[(tId - level1_num)];
		}
		else if (tId >= level0_num)
		{
			total_nodes[tId] = level1_nodes[(tId - level0_num)];
			total_value[tId] = level1_value[(tId - level0_num)];
			total_object[tId] = level1_object[(tId - level0_num)];
			total_normal[tId] = level1_normal[(tId - level0_num)];
		}
		else
		{
			total_nodes[tId] = level0_nodes[tId];
			total_value[tId] = level0_value[tId];
			total_object[tId] = level0_object[tId];
			total_normal[tId] = level0_normal[tId];
		}
		total_nodes[tId].setValueLocation(tId);
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::levelBottom(
		DArray<VoxelOctreeNode<Coord>>& grid0,
		DArray<Real>& grid0_value,
		DArray<Coord>& grid0_object,
		DArray<Coord>& grid0_normal,
		std::shared_ptr<TriangleSet<TDataType>> triSet,
		Coord m_origin,
		int m_nx,
		int m_ny,
		int m_nz,
		int padding,
		int& m_level0,
		Real m_dx)
	{
		// initialize data
		auto& triangles = triSet->getTriangles();
		auto& points = triSet->getPoints();
		auto& edges = triSet->getEdges();
		std::printf("the num of points edges surfaces is: %d %d %d \n", points.size(), edges.size(), triangles.size());

		triSet->updateTriangle2Edge();
		auto& tri2edg = triSet->getTriangle2Edge();

		DArray<Coord> edgeNormal, vertexNormal;
		triSet->updateEdgeNormal(edgeNormal);
		triSet->updateAngleWeightedVertexNormal(vertexNormal);

		DArray<int> data_count;
		data_count.resize(triangles.size());

		//数一下level_0中active grid的数目
		cuExecute(data_count.size(),
			SO_SurfaceCount,
			data_count,
			triangles,
			points,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz,
			padding);

		int grid_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), data_count.begin());

		DArray<PositionNode> grid_buf;
		grid_buf.resize(grid_num);

		//将level_0中的active grid取出
		cuExecute(data_count.size(),
			SO_SurfaceInit,
			grid_buf,
			data_count,
			triangles,
			points,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz,
			padding);

		thrust::sort(thrust::device, grid_buf.begin(), grid_buf.begin() + grid_buf.size(), PositionCmp());

		data_count.resize(grid_num);
		data_count.reset();
		//数不重复的grid
		cuExecute(data_count.size(),
			SO_CountNonRepeatedPosition,
			data_count,
			grid_buf);

		int grid0_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), data_count.begin());

		DArray<IndexNode> xIndex, yIndex, zIndex;
		xIndex.resize(grid0_num);
		yIndex.resize(grid0_num);
		zIndex.resize(grid0_num);
		grid0.resize(grid0_num);
		grid0_value.resize(grid0_num);
		grid0_object.resize(grid0_num);
		grid0_normal.resize(grid0_num);
		//去重
		cuExecute(data_count.size(),
			SO_FetchNonRepeatedPosition,
			grid0,
			grid0_value,
			grid0_object,
			grid0_normal,
			xIndex,
			yIndex,
			zIndex,
			grid_buf,
			data_count,
			tri2edg,
			edges,
			edgeNormal,
			vertexNormal,
			triangles,
			points,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);

		m_level0 = grid0_num;
		//std::printf("the grid num of level 0 is: %d \n", grid0_num);

		thrust::sort(thrust::device, xIndex.begin(), xIndex.begin() + xIndex.size(), IndexCmp());
		thrust::sort(thrust::device, yIndex.begin(), yIndex.begin() + yIndex.size(), IndexCmp());
		thrust::sort(thrust::device, zIndex.begin(), zIndex.begin() + zIndex.size(), IndexCmp());

		cuExecute(grid0_num,
			SO_UpdateBottomNeighbors,
			grid0,
			xIndex,
			yIndex,
			zIndex,
			m_nx,
			m_ny,
			m_nz);

		xIndex.clear();
		yIndex.clear();
		zIndex.clear();
		data_count.clear();
		grid_buf.clear();
		edgeNormal.clear();
		vertexNormal.clear();
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::levelMiddle(
		DArray<VoxelOctreeNode<Coord>>& grid1,
		DArray<Real>& grid1_value,
		DArray<Coord>& grid1_object,
		DArray<Coord>& grid1_normal,
		DArray<VoxelOctreeNode<Coord>>& grid0,
		DArray<Coord>& grid0_object,
		DArray<Coord>& grid0_normal,
		Coord m_origin,
		Level multi_level,
		int m_nx,
		int m_ny,
		int m_nz,
		Real m_dx)
	{
		Real coef = Real(pow(Real(2), int(multi_level)));
		int up_nx = m_nx / coef;
		int up_ny = m_ny / coef;
		int up_nz = m_nz / coef;

		int grid0_num = grid0.size();
		DArray<VoxelOctreeNode<Coord>> grid_buf1;
		DArray<Real> grid_value_buf1;
		DArray<Coord> grid_object_buf1, grid_normal_buf1;
		grid_buf1.resize(grid0_num);
		grid_value_buf1.resize(grid0_num);
		grid_object_buf1.resize(grid0_num);
		grid_normal_buf1.resize(grid0_num);
		//生成父节点及其兄弟节点
		cuExecute(grid0_num,
			SO_ComputeUpLevelGrids,
			grid_buf1,
			grid_value_buf1,
			grid_object_buf1,
			grid_normal_buf1,
			grid0,
			grid0_object,
			grid0_normal,
			m_origin,
			m_dx);

		thrust::sort(thrust::device, grid_buf1.begin(), grid_buf1.begin() + grid_buf1.size(), NodeCmp<Coord>());

		DArray<int> data_count;
		data_count.resize(grid0_num);
		data_count.reset();
		//数不重复的grid
		cuExecute(data_count.size(),
			SO_CountNonRepeatedGrids,
			data_count,
			grid_buf1,
			grid_value_buf1);

		int grid1_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), data_count.begin());

		DArray<IndexNode> xIndex, yIndex, zIndex;
		xIndex.resize(grid1_num);
		yIndex.resize(grid1_num);
		zIndex.resize(grid1_num);
		grid1.resize(grid1_num);
		grid1_value.resize(grid1_num);
		grid1_object.resize(grid1_num);
		grid1_normal.resize(grid1_num);
		//去重
		cuExecute(data_count.size(),
			SO_FetchNonRepeatedGrids,
			grid1,
			grid1_value,
			grid1_object,
			grid1_normal,
			xIndex,
			yIndex,
			zIndex,
			grid_buf1,
			grid_value_buf1,
			grid_object_buf1,
			grid_normal_buf1,
			data_count,
			up_nx,
			up_ny,
			up_nz);

		thrust::sort(thrust::device, xIndex.begin(), xIndex.begin() + xIndex.size(), IndexCmp());
		thrust::sort(thrust::device, yIndex.begin(), yIndex.begin() + yIndex.size(), IndexCmp());
		thrust::sort(thrust::device, zIndex.begin(), zIndex.begin() + zIndex.size(), IndexCmp());

		//FIM更新
		for (int i = 0; i < 3; i++)
		{
			cuExecute(grid1_num,
				SO_FIMUpLevelGrids,
				grid1,
				grid1_value,
				xIndex,
				yIndex,
				zIndex,
				grid1_object,
				grid1_normal,
				up_nx,
				up_ny,
				up_nz);
		}

		xIndex.clear();
		yIndex.clear();
		zIndex.clear();
		grid_buf1.clear();
		grid_value_buf1.clear();
		grid_object_buf1.clear();
		grid_normal_buf1.clear();
		data_count.clear();
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::levelTop(
		DArray<VoxelOctreeNode<Coord>>& grid2,
		DArray<Real>& grid2_value,
		DArray<Coord>& grid2_object,
		DArray<Coord>& grid2_normal,
		DArray<VoxelOctreeNode<Coord>>& grid1,
		DArray<Coord>& grid1_object,
		DArray<Coord>& grid1_normal,
		Coord m_origin,
		Level multi_level,
		int m_nx,
		int m_ny,
		int m_nz,
		Real m_dx)
	{
		Real coef = Real(pow(Real(2), int(multi_level)));
		int up_nx = m_nx / coef;
		int up_ny = m_ny / coef;
		int up_nz = m_nz / coef;
		Real dx_top = m_dx * coef;
		int grids_num = up_nx * up_ny * up_nz;

		grid2.resize(grids_num);
		grid2_value.resize(grids_num);
		grid2_object.resize(grids_num);
		grid2_normal.resize(grids_num);

		DArray<Coord> grid_object_buf, grid_normal_buf;
		grid_object_buf.resize(8 * grids_num);
		grid_normal_buf.resize(8 * grids_num);
		DArray<int> data_count;
		data_count.resize(grids_num);
		data_count.reset();
		int grid1_num = grid1.size();
		//生成topside节点:主要构建父子关系
		cuExecute(grid1_num,
			SO_ComputeTopLevelGrids,
			grid2,
			grid_object_buf,
			grid_normal_buf,
			data_count,
			grid1,
			grid1_object,
			grid1_normal,
			m_origin,
			dx_top,
			up_nx,
			up_ny,
			up_nz,
			multi_level);

		//更新topside节点中的值
		cuExecute(grids_num,
			SO_UpdateTopLevelGrids,
			grid2,
			grid2_value,
			grid2_object,
			grid2_normal,
			data_count,
			grid_object_buf,
			grid_normal_buf,
			m_origin,
			dx_top,
			up_nx,
			up_ny,
			up_nz,
			multi_level);

		Reduction<int> reduce;
		DArray<int> data_count_temp;
		int total_num = reduce.accumulate(data_count.begin(), data_count.size());
		while (total_num < (grids_num))
		{
			grid_object_buf.assign(grid2_object);
			grid_normal_buf.assign(grid2_normal);
			data_count_temp.assign(data_count);
			//topside节点中的值FIM更新
			cuExecute(grids_num,
				SO_FIMComputeTopLevelGrids,
				grid2,
				grid2_value,
				grid2_object,
				grid2_normal,
				data_count,
				grid_object_buf,
				grid_normal_buf,
				data_count_temp,
				up_nx,
				up_ny,
				up_nz);

			total_num = reduce.accumulate(data_count.begin(), data_count.size());
		}
		data_count_temp.clear();
		data_count.clear();
		grid_object_buf.clear();
		grid_normal_buf.clear();
	}


	template<typename TDataType>
	void VolumeHelper<TDataType>::levelCollection(
		DArray<VoxelOctreeNode<Coord>>& grids,
		DArray<Real>& grids_value,
		DArray<Coord>& grids_object,
		DArray<Coord>& grids_normal,
		DArray<VoxelOctreeNode<Coord>>& grid1,
		DArray<Real>& grid1_value,
		DArray<Coord>& grid1_object,
		DArray<Coord>& grid1_normal,
		int uplevel_num)
	{
		int grids_num_temp = grids.size() + grid1.size();
		DArray<VoxelOctreeNode<Coord>> grids_temp;
		DArray<Real> grids_value_temp;
		DArray<Coord> grids_object_temp, grids_normal_temp;
		grids_temp.resize(grids_num_temp);
		grids_value_temp.resize(grids_num_temp);
		grids_object_temp.resize(grids_num_temp);
		grids_normal_temp.resize(grids_num_temp);
		cuExecute(grids_num_temp,
			SO_CollectionGrids,
			grids_temp,
			grids_value_temp,
			grids_object_temp,
			grids_normal_temp,
			grids,
			grids_value,
			grids_object,
			grids_normal,
			grid1,
			grid1_value,
			grid1_object,
			grid1_normal,
			grids.size(),
			uplevel_num);
		grids.assign(grids_temp);
		grids_value.assign(grids_value_temp);
		grids_object.assign(grids_object_temp);
		grids_normal.assign(grids_normal_temp);
		grids_temp.clear();
		grids_value_temp.clear();
		grids_object_temp.clear();
		grids_normal_temp.clear();
	}

	template <typename Real, typename Coord>
	__global__ void SO_CollectionGridsTwo(
		DArray<VoxelOctreeNode<Coord>> total_nodes,
		DArray<Real> total_value,
		DArray<Coord> total_object,
		DArray<Coord> total_normal,
		DArray<VoxelOctreeNode<Coord>> level0_nodes,
		DArray<Real> level0_value,
		DArray<Coord> level0_object,
		DArray<Coord> level0_normal,
		DArray<VoxelOctreeNode<Coord>> levelT_nodes,
		DArray<Real> levelT_value,
		DArray<Coord> levelT_object,
		DArray<Coord> levelT_normal,
		int level0_num)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= total_nodes.size()) return;

		if (tId >= level0_num)
		{
			total_nodes[tId] = levelT_nodes[(tId - level0_num)];
			total_nodes[tId].setValueLocation(tId);
			for (int i = 0; i < 6; i++)
			{
				if (total_nodes[tId].m_neighbor[i] != EMPTY)
					total_nodes[tId].m_neighbor[i] += level0_num;
			}

			total_value[tId] = levelT_value[(tId - level0_num)];
			total_object[tId] = levelT_object[(tId - level0_num)];
			total_normal[tId] = levelT_normal[(tId - level0_num)];
		}
		else
		{
			total_nodes[tId] = level0_nodes[tId];
			total_nodes[tId].setValueLocation(tId);

			total_value[tId] = level0_value[tId];
			total_object[tId] = level0_object[tId];
			total_normal[tId] = level0_normal[tId];
		}
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::collectionGridsTwo(
		DArray<VoxelOctreeNode<Coord>>& total_nodes,
		DArray<Real>& total_value, 
		DArray<Coord>& total_object,
		DArray<Coord>& total_normal,
		DArray<VoxelOctreeNode<Coord>>& level0_nodes,
		DArray<Real>& level0_value,
		DArray<Coord>& level0_object,
		DArray<Coord>& level0_normal,
		DArray<VoxelOctreeNode<Coord>>& levelT_nodes,
		DArray<Real>& levelT_value,
		DArray<Coord>& levelT_object,
		DArray<Coord>& levelT_normal,
		int level0_num,
		int grid_total_num)
	{
		cuExecute(grid_total_num,
			SO_CollectionGridsTwo,
			total_nodes,
			total_value,
			total_object,
			total_normal,
			level0_nodes,
			level0_value,
			level0_object,
			level0_normal,
			levelT_nodes,
			levelT_value,
			levelT_object,
			levelT_normal,
			level0_nodes.size());
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::collectionGridsThree(
		DArray<VoxelOctreeNode<Coord>>& total_nodes, 
		DArray<Real>& total_value, 
		DArray<Coord>& total_object, 
		DArray<Coord>& total_normal,
		DArray<VoxelOctreeNode<Coord>>& level0_nodes, 
		DArray<Real>& level0_value, 
		DArray<Coord>& level0_object, 
		DArray<Coord>& level0_normal, 
		DArray<VoxelOctreeNode<Coord>>& level1_nodes, 
		DArray<Real>& level1_value, 
		DArray<Coord>& level1_object, 
		DArray<Coord>& level1_normal, 
		DArray<VoxelOctreeNode<Coord>>& levelT_nodes, 
		DArray<Real>& levelT_value, 
		DArray<Coord>& levelT_object, 
		DArray<Coord>& levelT_normal, 
		int level0_num, 
		int level1_num,
		int grid_total_num)
	{
		cuExecute(grid_total_num,
			SO_CollectionGridsThree,
			total_nodes,
			total_value,
			total_object,
			total_normal,
			level0_nodes,
			level0_value,
			level0_object,
			level0_normal,
			level1_nodes,
			level1_value,
			level1_object,
			level1_normal,
			levelT_nodes,
			levelT_value,
			levelT_object,
			levelT_normal,
			level0_nodes.size(),
			(level0_nodes.size() + level1_nodes.size()));
	}


	template<typename TDataType>
	void VolumeHelper<TDataType>::collectionGridsFour(
		DArray<VoxelOctreeNode<Coord>>& total_nodes, 
		DArray<Real>& total_value, 
		DArray<Coord>& total_object, 
		DArray<Coord>& total_normal, 
		DArray<VoxelOctreeNode<Coord>>& level0_nodes, 
		DArray<Real>& level0_value, 
		DArray<Coord>& level0_object, 
		DArray<Coord>& level0_normal, 
		DArray<VoxelOctreeNode<Coord>>& level1_nodes, 
		DArray<Real>& level1_value, 
		DArray<Coord>& level1_object, 
		DArray<Coord>& level1_normal, 
		DArray<VoxelOctreeNode<Coord>>& level2_nodes,
		DArray<Real>& level2_value, 
		DArray<Coord>& level2_object, 
		DArray<Coord>& level2_normal, 
		DArray<VoxelOctreeNode<Coord>>& levelT_nodes, 
		DArray<Real>& levelT_value, 
		DArray<Coord>& levelT_object, 
		DArray<Coord>& levelT_normal, 
		int level0_num, 
		int level1_num, 
		int level2_num,
		int grid_total_num)
	{
		cuExecute(grid_total_num,
			SO_CollectionGridsFour,
			total_nodes,
			total_value,
			total_object,
			total_normal,
			level0_nodes,
			level0_value,
			level0_object,
			level0_normal,
			level1_nodes,
			level1_value,
			level1_object,
			level1_normal,
			level2_nodes,
			level2_value,
			level2_object,
			level2_normal,
			levelT_nodes,
			levelT_value,
			levelT_object,
			levelT_normal,
			level0_nodes.size(),
			(level0_nodes.size() + level1_nodes.size()),
			(level0_nodes.size() + level1_nodes.size() + level2_nodes.size()));
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::collectionGridsFive(
		DArray<VoxelOctreeNode<Coord>>& total_nodes,
		DArray<Real>& total_value,
		DArray<Coord>& total_object,
		DArray<Coord>& total_normal,
		DArray<VoxelOctreeNode<Coord>>& level0_nodes,
		DArray<Real>& level0_value,
		DArray<Coord>& level0_object,
		DArray<Coord>& level0_normal,
		DArray<VoxelOctreeNode<Coord>>& level1_nodes,
		DArray<Real>& level1_value,
		DArray<Coord>& level1_object,
		DArray<Coord>& level1_normal,
		DArray<VoxelOctreeNode<Coord>>& level2_nodes,
		DArray<Real>& level2_value,
		DArray<Coord>& level2_object,
		DArray<Coord>& level2_normal,
		DArray<VoxelOctreeNode<Coord>>& level3_nodes,
		DArray<Real>& level3_value,
		DArray<Coord>& level3_object,
		DArray<Coord>& level3_normal,
		DArray<VoxelOctreeNode<Coord>>& levelT_nodes,
		DArray<Real>& levelT_value,
		DArray<Coord>& levelT_object,
		DArray<Coord>& levelT_normal,
		int level0_num,
		int level1_num,
		int level2_num,
		int level3_num,
		int grid_total_num)
	{
		cuExecute(grid_total_num,
			SO_CollectionGridsFive,
			total_nodes,
			total_value,
			total_object,
			total_normal,
			level0_nodes,
			level0_value,
			level0_object,
			level0_normal,
			level1_nodes,
			level1_value,
			level1_object,
			level1_normal,
			level2_nodes,
			level2_value,
			level2_object,
			level2_normal,
			level3_nodes,
			level3_value,
			level3_object,
			level3_normal,
			levelT_nodes,
			levelT_value,
			levelT_object,
			levelT_normal,
			level0_nodes.size(),
			(level0_nodes.size() + level1_nodes.size()),
			(level0_nodes.size() + level1_nodes.size() + level2_nodes.size()),
			(level0_nodes.size() + level1_nodes.size() + level2_nodes.size() + level3_nodes.size()));
	}


	DYN_FUNC static void kernel3(Real& val, Real val_x)
	{
		if (std::abs(val_x) < 1)
			val = ((0.5*abs(val_x)*abs(val_x)*abs(val_x)) - (abs(val_x)*abs(val_x)) + (2.0f / 3.0f));
		else if (std::abs(val_x) < 2)
			val = (1.0f / 6.0f)*(2.0f - (std::abs(val_x)))*(2.0f - (std::abs(val_x)))*(2.0f - (std::abs(val_x)));
		else
			val = 0.0f;
	}

	template <typename Coord>
	__global__ void SO_NodesInitialTopology(
		DArray<PositionNode> nodes,
		DArray<Coord> pos,
		DArray<VoxelOctreeNode<Coord>> nodes_a,
		DArray<VoxelOctreeNode<Coord>> nodes_b,
		int off_ax,
		int off_ay,
		int off_az,
		int off_bx,
		int off_by,
		int off_bz,
		int level0_a)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		if (tId < level0_a)
		{
			OcIndex gnx0, gny0, gnz0, gnx1, gny1, gnz1;
			OcKey old_key = nodes_a[tId].key();
			RecoverFromMortonCode(old_key, gnx0, gny0, gnz0);

			gnx1 = gnx0 + off_ax;
			gny1 = gny0 + off_ay;
			gnz1 = gnz0 + off_az;
			OcKey new_key = CalculateMortonCode(gnx1, gny1, gnz1);

			nodes[tId] = PositionNode(tId, new_key);
			pos[tId] = nodes_a[tId].position();
		}
		else
		{
			OcIndex gnx0, gny0, gnz0, gnx1, gny1, gnz1;
			OcKey old_key = nodes_b[tId - level0_a].key();
			RecoverFromMortonCode(old_key, gnx0, gny0, gnz0);

			gnx1 = gnx0 + off_bx;
			gny1 = gny0 + off_by;
			gnz1 = gnz0 + off_bz;
			OcKey new_key = CalculateMortonCode(gnx1, gny1, gnz1);

			nodes[tId] = PositionNode(tId, new_key);
			pos[tId] = nodes_b[tId - level0_a].position();
		}
	}

	template <typename Real>
	__global__ void SO_CountNonRepeatedUnion(
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<Real> sdf_a,
		DArray<Real> sdf_b,
		DArray<Real> value_a,
		DArray<Real> value_b,
		int level0_a)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || (nodes[tId].position_index != nodes[tId - 1].position_index))
		{
			if (tId < counter.size() - 1 && (nodes[tId].position_index == nodes[tId + 1].position_index))
			{
				Real sdf_v1, sdf_v2;
				if (nodes[tId].surface_index < level0_a)
				{
					sdf_v1 = sdf_a[nodes[tId].surface_index];
					sdf_v2 = sdf_b[nodes[tId + 1].surface_index - level0_a];
				}
				else
				{
					sdf_v1 = sdf_b[nodes[tId].surface_index - level0_a];
					sdf_v2 = sdf_a[nodes[tId + 1].surface_index];
				}

				if (sdf_v1 < sdf_v2)
					counter[tId] = 1;
				else
					counter[tId + 1] = 1;
			}
			else if (tId == counter.size() - 1 || (nodes[tId].position_index != nodes[tId + 1].position_index))
			{
				if (nodes[tId].surface_index < level0_a)
				{
					if (value_b[nodes[tId].surface_index] > 0)
						counter[tId] = 1;
				}
				else
				{
					if (value_a[nodes[tId].surface_index] > 0)
						counter[tId] = 1;
				}
			}
		}
	}

	template <typename Real>
	__global__ void SO_CountNonRepeatedIntersection(
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<Real> sdf_a,
		DArray<Real> sdf_b,
		DArray<Real> value_a,
		DArray<Real> value_b,
		int level0_a)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || (nodes[tId].position_index != nodes[tId - 1].position_index))
		{
			if (tId < counter.size() - 1 && (nodes[tId].position_index == nodes[tId + 1].position_index))
			{
				Real sdf_v1, sdf_v2;
				if (nodes[tId].surface_index < level0_a)
				{
					sdf_v1 = sdf_a[nodes[tId].surface_index];
					sdf_v2 = sdf_b[nodes[tId + 1].surface_index - level0_a];
				}
				else
				{
					sdf_v1 = sdf_b[nodes[tId].surface_index - level0_a];
					sdf_v2 = sdf_a[nodes[tId + 1].surface_index];
				}

				if (sdf_v1 > sdf_v2)
					counter[tId] = 1;
				else
					counter[tId + 1] = 1;
			}
			else if (tId == counter.size() - 1 || (nodes[tId].position_index != nodes[tId + 1].position_index))
			{
				if (nodes[tId].surface_index < level0_a)
				{
					if (value_b[nodes[tId].surface_index] < 0)
						counter[tId] = 1;
				}
				else
				{
					if (value_a[nodes[tId].surface_index] < 0)
						counter[tId] = 1;
				}
			}
		}
	}

	template <typename Real>
	__global__ void SO_CountNonRepeatedSubtractionA(
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<Real> sdf_a,
		DArray<Real> sdf_b,
		DArray<Real> value_a,
		DArray<Real> value_b,
		int level0_a)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || (nodes[tId].position_index != nodes[tId - 1].position_index))
		{
			if (tId < counter.size() - 1 && (nodes[tId].position_index == nodes[tId + 1].position_index))
			{
				Real sdf_v1, sdf_v2;
				if (nodes[tId].surface_index < level0_a)
				{
					sdf_v1 = sdf_a[nodes[tId].surface_index];
					sdf_v2 = sdf_b[nodes[tId + 1].surface_index - level0_a];
					if (sdf_v1 > -sdf_v2)
						counter[tId] = 1;
					else
						counter[tId + 1] = 1;
				}
				else
				{
					sdf_v1 = sdf_b[nodes[tId].surface_index - level0_a];
					sdf_v2 = sdf_a[nodes[tId + 1].surface_index];
					if (-sdf_v1 > sdf_v2)
						counter[tId] = 1;
					else
						counter[tId + 1] = 1;
				}
			}
			else if (tId == counter.size() - 1 || (nodes[tId].position_index != nodes[tId + 1].position_index))
			{
				if (nodes[tId].surface_index < level0_a)
				{
					if (value_b[nodes[tId].surface_index] > 0)
						counter[tId] = 1;
				}
				else
				{
					if (value_a[nodes[tId].surface_index] < 0)
						counter[tId] = 1;
				}
			}
		}
	}

	template <typename Real>
	__global__ void SO_CountNonRepeatedSubtractionB(
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<Real> sdf_a,
		DArray<Real> sdf_b,
		DArray<Real> value_a,
		DArray<Real> value_b,
		int level0_a)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || (nodes[tId].position_index != nodes[tId - 1].position_index))
		{
			if (tId < counter.size() - 1 && (nodes[tId].position_index == nodes[tId + 1].position_index))
			{
				Real sdf_v1, sdf_v2;
				if (nodes[tId].surface_index < level0_a)
				{
					sdf_v1 = sdf_a[nodes[tId].surface_index];
					sdf_v2 = sdf_b[nodes[tId + 1].surface_index - level0_a];
					if (-sdf_v1 > sdf_v2)
						counter[tId] = 1;
					else
						counter[tId + 1] = 1;
				}
				else
				{
					sdf_v1 = sdf_b[nodes[tId].surface_index - level0_a];
					sdf_v2 = sdf_a[nodes[tId + 1].surface_index];
					if (sdf_v1 > -sdf_v2)
						counter[tId] = 1;
					else
						counter[tId + 1] = 1;
				}
			}
			else if (tId == counter.size() - 1 || (nodes[tId].position_index != nodes[tId + 1].position_index))
			{
				if (nodes[tId].surface_index < level0_a)
				{
					if (value_b[nodes[tId].surface_index] < 0)
						counter[tId] = 1;
				}
				else
				{
					if (value_a[nodes[tId].surface_index] > 0)
						counter[tId] = 1;
				}
			}
		}
	}

	//注意counter中第i+1个元素存的是0-i元素的和
	template <typename Real, typename Coord>
	__global__ void SO_FetchNonRepeatedGridsTopology(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<PositionNode> all_nodes,
		DArray<int> counter,
		DArray<VoxelOctreeNode<Coord>> nodes_a,
		DArray<Real> nodes_value_a,
		DArray<Coord> nodes_object_a,
		DArray<Coord> nodes_normal_a,
		DArray<VoxelOctreeNode<Coord>> nodes_b,
		DArray<Real> nodes_value_b,
		DArray<Coord> nodes_object_b,
		DArray<Coord> nodes_normal_b,
		int level0_a,
		int op,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId < (counter.size() - 1) && counter[tId] != counter[tId + 1])
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(all_nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			VoxelOctreeNode<Coord> mc((Level)0, gnx, gny, gnz, pos);
			mc.setValueLocation(counter[tId]);

			if (all_nodes[tId].surface_index < level0_a)
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_a[all_nodes[tId].surface_index];
				nodes_object[counter[tId]] = nodes_object_a[all_nodes[tId].surface_index];
				//if (op == VolumeOctreeUnion<DataType3f>::BooleanOperation::SUBTRACTION_SETB)
				//	nodes_normal[counter[tId]] = -nodes_normal_a[all_nodes[tId].surface_index];
				//else
					nodes_normal[counter[tId]] = nodes_normal_a[all_nodes[tId].surface_index];
			}
			else
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_b[all_nodes[tId].surface_index - level0_a];
				nodes_object[counter[tId]] = nodes_object_b[all_nodes[tId].surface_index - level0_a];
				if (op == 2)
					nodes_normal[counter[tId]] = -nodes_normal_b[all_nodes[tId].surface_index - level0_a];
				else
					nodes_normal[counter[tId]] = nodes_normal_b[all_nodes[tId].surface_index - level0_a];
			}

			x_index[counter[tId]] = IndexNode((gnz*nx_*ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx*ny_*nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny*nz_*nx_ + gnx * nz_ + gnz), counter[tId]);
		}
		else if (tId == (counter.size() - 1) && (counter[tId] < nodes.size()))
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(all_nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			VoxelOctreeNode<Coord> mc((Level)0, gnx, gny, gnz, pos);
			mc.setValueLocation(counter[tId]);

			if (all_nodes[tId].surface_index < level0_a)
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_a[all_nodes[tId].surface_index];
				nodes_object[counter[tId]] = nodes_object_a[all_nodes[tId].surface_index];
				//if (op == VolumeOctreeUnion<DataType3f>::BooleanOperation::SUBTRACTION_SETB)
				//	nodes_normal[counter[tId]] = -nodes_normal_a[all_nodes[tId].surface_index];
				//else
					nodes_normal[counter[tId]] = nodes_normal_a[all_nodes[tId].surface_index];
			}
			else
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_b[all_nodes[tId].surface_index - level0_a];
				nodes_object[counter[tId]] = nodes_object_b[all_nodes[tId].surface_index - level0_a];
				if (op == 2)
					nodes_normal[counter[tId]] = -nodes_normal_b[all_nodes[tId].surface_index - level0_a];
				else
					nodes_normal[counter[tId]] = nodes_normal_b[all_nodes[tId].surface_index - level0_a];
			}

			x_index[counter[tId]] = IndexNode((gnz*nx_*ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx*ny_*nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny*nz_*nx_ + gnx * nz_ + gnz), counter[tId]);
		}
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::finestLevelBoolean(
		DArray<VoxelOctreeNode<Coord>>& grid0,
		DArray<Real>& grid0_value,
		DArray<Coord>& grid0_object,
		DArray<Coord>& grid0_normal,
		DArray<VoxelOctreeNode<Coord>>& sdfOctreeNode_a,
		DArray<Real>& sdfValue_a,
		DArray<Coord>& object_a,
		DArray<Coord>& normal_a,
		DArray<VoxelOctreeNode<Coord>>& sdfOctreeNode_b,
		DArray<Real>& sdfValue_b,
		DArray<Coord>& object_b,
		DArray<Coord>& normal_b,
		std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_a,
		std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_b,
		int offset_ax,
		int offset_ay,
		int offset_az,
		int offset_bx,
		int offset_by,
		int offset_bz,
		int level0_a,
		int level0_b,
		int& m_level0,
		Coord m_origin,
		Real m_dx,
		int m_nx,
		int m_ny,
		int m_nz,
		int boolean)//boolean=0:union, boolean=1:intersection, boolean=2:subtraction
	{

		int level0_num = level0_a + level0_b;

		DArray<PositionNode> grid_buf;
		grid_buf.resize(level0_num);
		DArray<Coord> grid_pos;
		grid_pos.resize(level0_num);
		//将level0放到一起
		cuExecute(level0_num,
			SO_NodesInitialTopology,
			grid_buf,
			grid_pos,
			sdfOctreeNode_a,
			sdfOctreeNode_b,
			offset_ax,
			offset_ay,
			offset_az,
			offset_bx,
			offset_by,
			offset_bz,
			level0_a);

		thrust::sort(thrust::device, grid_buf.begin(), grid_buf.begin() + grid_buf.size(), PositionCmp());

		DArray<Real> pos_value_a, pos_value_b;
		DArray<Coord> pos_normal_a, pos_normal_b;
		sdfOctree_a->getSignDistanceMLS(grid_pos, pos_value_a, pos_normal_a);
		sdfOctree_b->getSignDistanceMLS(grid_pos, pos_value_b, pos_normal_b);
		pos_normal_a.clear();
		pos_normal_b.clear();

		DArray<int> data_count;
		data_count.resize(level0_num);
		data_count.reset();
		//数不重复的grid
		if (boolean == 0)
		{
			cuExecute(data_count.size(),
				SO_CountNonRepeatedUnion,
				data_count,
				grid_buf,
				sdfValue_a,
				sdfValue_b,
				pos_value_a,
				pos_value_b,
				level0_a);
		}
		else if (boolean == 1)
		{
			cuExecute(data_count.size(),
				SO_CountNonRepeatedIntersection,
				data_count,
				grid_buf,
				sdfValue_a,
				sdfValue_b,
				pos_value_a,
				pos_value_b,
				level0_a);
		}
		else if (boolean == 2)
		{
			cuExecute(data_count.size(),
				SO_CountNonRepeatedSubtractionA,
				data_count,
				grid_buf,
				sdfValue_a,
				sdfValue_b,
				pos_value_a,
				pos_value_b,
				level0_a);
		}
		int grid0_num = thrust::reduce(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, data_count.begin(), data_count.begin() + data_count.size(), data_count.begin());

		if (grid0_num == 0) { return; }

		m_level0 = grid0_num;

		DArray<IndexNode> xIndex, yIndex, zIndex;
		xIndex.resize(grid0_num);
		yIndex.resize(grid0_num);
		zIndex.resize(grid0_num);
		grid0.resize(grid0_num);
		grid0_value.resize(grid0_num);
		grid0_object.resize(grid0_num);
		grid0_normal.resize(grid0_num);
		//去重
		cuExecute(data_count.size(),
			SO_FetchNonRepeatedGridsTopology,
			grid0,
			grid0_value,
			grid0_object,
			grid0_normal,
			xIndex,
			yIndex,
			zIndex,
			grid_buf,
			data_count,
			sdfOctreeNode_a,
			sdfValue_a,
			object_a,
			normal_a,
			sdfOctreeNode_b,
			sdfValue_b,
			object_b,
			normal_b,
			level0_a,
			boolean,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);

		thrust::sort(thrust::device, xIndex.begin(), xIndex.begin() + xIndex.size(), IndexCmp());
		thrust::sort(thrust::device, yIndex.begin(), yIndex.begin() + yIndex.size(), IndexCmp());
		thrust::sort(thrust::device, zIndex.begin(), zIndex.begin() + zIndex.size(), IndexCmp());

		cuExecute(grid0_num,
			SO_UpdateBottomNeighbors,
			grid0,
			xIndex,
			yIndex,
			zIndex,
			m_nx,
			m_ny,
			m_nz);

		xIndex.clear();
		yIndex.clear();
		zIndex.clear();
		data_count.clear();
		grid_buf.clear();
		grid_pos.clear();
		pos_value_a.clear();
		pos_value_b.clear();
	}

	template <typename Real, typename Coord>
	GPU_FUNC int SO_ComputeGrid(
		int& nx_lo,
		int& ny_lo,
		int& nz_lo,
		int& nx_hi,
		int& ny_hi,
		int& nz_hi,
		VoxelOctreeNode<Coord>& node,
		Coord p0,
		Coord p1,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		nx_lo = std::floor((p0[0] - origin_[0]) / dx_);
		ny_lo = std::floor((p0[1] - origin_[1]) / dx_);
		nz_lo = std::floor((p0[2] - origin_[2]) / dx_);

		nx_hi = std::ceil((p1[0] - origin_[0]) / dx_);
		ny_hi = std::ceil((p1[1] - origin_[1]) / dx_);
		nz_hi = std::ceil((p1[2] - origin_[2]) / dx_);

		if ((node.m_neighbor[0] == EMPTY) && ((nx_lo % 2) != 0)) nx_lo--;
		if ((node.m_neighbor[2] == EMPTY) && ((ny_lo % 2) != 0)) ny_lo--;
		if ((node.m_neighbor[4] == EMPTY) && ((nz_lo % 2) != 0)) nz_lo--;
		if ((node.m_neighbor[1] == EMPTY) && ((nx_hi % 2) != 1)) nx_hi++;
		if ((node.m_neighbor[3] == EMPTY) && ((ny_hi % 2) != 1)) ny_hi++;
		if ((node.m_neighbor[5] == EMPTY) && ((nz_hi % 2) != 1)) nz_hi++;

		return (nz_hi - nz_lo + 1) * (ny_hi - ny_lo + 1) * (nx_hi - nx_lo + 1);
	}

	template <typename Real, typename Coord>
	__global__ void SO_CountReconstructedGrids(
		DArray<int> counter,
		DArray<VoxelOctreeNode<Coord>> nodes,
		Coord origin_,
		Real dx_,
		Real olddx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		Coord p0 = nodes[tId].position() - 0.5*olddx_;
		Coord p1 = nodes[tId].position() + 0.5*olddx_;

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		counter[tId] = SO_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, nodes[tId], p0, p1, origin_, dx_, nx_, ny_, nz_);
	}

	template <typename Real, typename Coord>
	__global__ void SO_FetchReconstructedGrids(
		DArray<PositionNode> nodes_buf,
		DArray<int> counter,
		DArray<VoxelOctreeNode<Coord>> nodes,
		Coord origin_,
		Real dx_,
		Real olddx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		Coord p0 = nodes[tId].position() - 0.5*olddx_;
		Coord p1 = nodes[tId].position() + 0.5*olddx_;

		int nx_lo;
		int ny_lo;
		int nz_lo;

		int nx_hi;
		int ny_hi;
		int nz_hi;

		int num = SO_ComputeGrid(nx_lo, ny_lo, nz_lo, nx_hi, ny_hi, nz_hi, nodes[tId], p0, p1, origin_, dx_, nx_, ny_, nz_);

		if (num > 0)
		{
			int acc_num = 0;
			for (int k = nz_lo; k <= nz_hi; k++) {
				for (int j = ny_lo; j <= ny_hi; j++) {
					for (int i = nx_lo; i <= nx_hi; i++)
					{
						OcKey index = CalculateMortonCode(i, j, k);
						nodes_buf[counter[tId] + acc_num] = PositionNode(tId, index);

						acc_num++;
					}
				}
			}
		}
	}

	__global__ void SO_CountNoRepeatedReconstructedGrids(
		DArray<int> counter,
		DArray<PositionNode> nodes)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || nodes[tId].position_index != nodes[tId - 1].position_index)
		{
			counter[tId] = 1;
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_FetchNoRepeatedReconstructedGrids(
		DArray<PositionNode> nodes_new,
		DArray<Real> nodes_sdf,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<VoxelOctreeNode<Coord>> nodes_old,
		DArray<Coord> object,
		DArray<Coord> normal,
		Coord origin_,
		Real dx_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId == 0 || nodes[tId].position_index != nodes[tId - 1].position_index)
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);
			Coord pos_old(nodes_old[nodes[tId].surface_index].position());

			//直接取最近的投影点
			Real dist = (pos - object[nodes[tId].surface_index]).norm();
			int index = tId;
			int num = 1;
			while (nodes[tId + num].position_index == nodes[tId].position_index)
			{
				pos_old = nodes_old[nodes[tId + num].surface_index].position();

				Real dist_num = (pos - object[nodes[tId + num].surface_index]).norm();
				if (dist_num < dist)
				{
					index = tId + num;
					dist = dist_num;
				}
				num++;
			}
			nodes_new[counter[tId]] = nodes[index];
			nodes_object[counter[tId]] = object[nodes[index].surface_index];
			nodes_normal[counter[tId]] = normal[nodes[index].surface_index];
			Real sign = (pos - nodes_object[counter[tId]]).dot(nodes_normal[counter[tId]]) < Real(0) ? Real(-1) : Real(1);
			nodes_sdf[counter[tId]] = sign * dist;


			////直接求平均
			//Coord object_sum(object[nodes[tId].surface_index]);
			//Coord normal_sum(normal[nodes[tId].surface_index]);
			//int num = 1;
			//while (nodes[tId + num].position_index == nodes[tId].position_index)
			//{
			//	object_sum += object[nodes[tId + num].surface_index];
			//	normal_sum += normal[nodes[tId + num].surface_index];
			//	num++;
			//}
			//nodes_new[counter[tId]] = nodes[tId];
			//nodes_object[counter[tId]] = object_sum / num;
			//nodes_normal[counter[tId]] = normal_sum / num;
			//Real sign = (pos - nodes_object[counter[tId]]).dot(nodes_normal[counter[tId]]) < Real(0) ? Real(-1) : Real(1);
			//Real dist = (pos - nodes_object[counter[tId]]).norm();
			//nodes_sdf[counter[tId]] = sign * dist;

			////线性插值
			//int nnx = 0, nny = 0, nnz = 0;
			//Coord object_linter, normal_linter;
			//Coord obj_old = object[nodes[tId].surface_index];
			//Coord nor_old = normal[nodes[tId].surface_index];
			//int num = 1;
			//while (nodes[tId + num].position_index == nodes[tId].position_index)
			//{
			//	Coord pos_num(nodes_old[nodes[tId + num].surface_index].position());
			//	Coord obj_num(object[nodes[tId + num].surface_index]);
			//	Coord nor_num(normal[nodes[tId + num].surface_index]);
			//	if (nnx == 0 && abs(pos_num[0] - pos_old[0]) > (10 * REAL_EPSILON))
			//	{
			//		object_linter[0] = obj_num[0] * (pos[0] - pos_old[0]) / (pos_num[0] - pos_old[0]) + obj_old[0] * (pos_num[0] - pos[0]) / (pos_num[0] - pos_old[0]);
			//		normal_linter[0] = nor_num[0] * (pos[0] - pos_old[0]) / (pos_num[0] - pos_old[0]) + nor_old[0] * (pos_num[0] - pos[0]) / (pos_num[0] - pos_old[0]);
			//		nnx = 1;
			//	}
			//	if (nny == 0 && abs(pos_num[1] - pos_old[1]) > (10 * REAL_EPSILON))
			//	{
			//		object_linter[1] = obj_num[1] * (pos[1] - pos_old[1]) / (pos_num[1] - pos_old[1]) + obj_old[1] * (pos_num[1] - pos[1]) / (pos_num[1] - pos_old[1]);
			//		normal_linter[1] = nor_num[1] * (pos[1] - pos_old[1]) / (pos_num[1] - pos_old[1]) + nor_old[1] * (pos_num[1] - pos[1]) / (pos_num[1] - pos_old[1]);
			//		nny = 1;
			//	}
			//	if (nnz == 0 && abs(pos_num[2] - pos_old[2]) > (10 * REAL_EPSILON))
			//	{
			//		object_linter[2] = obj_num[2] * (pos[2] - pos_old[2]) / (pos_num[2] - pos_old[2]) + obj_old[2] * (pos_num[2] - pos[2]) / (pos_num[2] - pos_old[2]);
			//		normal_linter[2] = nor_num[2] * (pos[2] - pos_old[2]) / (pos_num[2] - pos_old[2]) + nor_old[2] * (pos_num[2] - pos[2]) / (pos_num[2] - pos_old[2]);
			//		nnz = 1;
			//	}
			//	if (nnx == 1 && (nny == 1 && nnz == 1))
			//	{
			//		break;
			//	}
			//	num++;
			//}
			//nodes_new[counter[tId]] = nodes[tId];
			//nodes_object[counter[tId]] = object_linter;
			//nodes_normal[counter[tId]] = normal_linter;
			//Real sign = (pos - nodes_object[counter[tId]]).dot(nodes_normal[counter[tId]]) < Real(0) ? Real(-1) : Real(1);
			//Real dist = (pos - nodes_object[counter[tId]]).norm();
			//nodes_sdf[counter[tId]] = sign * dist;

			//printf("Boolean neighbor: %d %d \n", tId, num);
		}
	}

	//根据现有的最底层网格进行插值（采用权重与距离相关的非线性插值）
	template <typename Real, typename Coord>
	__global__ void SO_FetchAndInterpolateReconstructedGrids(
		DArray<PositionNode> nodes_new,
		DArray<Real> nodes_sdf,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<VoxelOctreeNode<Coord>> nodes_old,
		DArray<Coord> object,
		DArray<Coord> normal,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		if (tId == 0 || nodes[tId].position_index != nodes[tId - 1].position_index)
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);
			Coord pos_old(nodes_old[nodes[tId].surface_index].position());

			int multiple = 1.5;

			Real dist = (pos - pos_old).norm();
			Real weight_sum(1.0f), weight(1.0f);
			kernel3(weight_sum, dist / (multiple*dx_));

			int num = 1;
			while (nodes[tId + num].position_index == nodes[tId].position_index)
			{
				pos_old = nodes_old[nodes[tId + num].surface_index].position();

				dist = (pos - pos_old).norm();
				kernel3(weight, dist / (multiple*dx_));

				weight_sum += weight;

				num++;
			}

			Coord pobject(object[nodes[tId].surface_index]);
			Coord pnormal(normal[nodes[tId].surface_index]);

			if (weight_sum > 10 * REAL_EPSILON)
			{
				pos_old = nodes_old[nodes[tId].surface_index].position();
				dist = (pos - pos_old).norm();
				kernel3(weight, dist / (multiple*dx_));

				pobject = pobject * (weight / weight_sum);
				pnormal = pnormal * (weight / weight_sum);

				num = 1;
				while (nodes[tId + num].position_index == nodes[tId].position_index)
				{
					pos_old = nodes_old[nodes[tId + num].surface_index].position();
					dist = (pos - pos_old).norm();
					kernel3(weight, dist / (multiple*dx_));
					pobject += ((weight / weight_sum)*object[(nodes[tId + num].surface_index)]);
					pnormal += ((weight / weight_sum)*normal[(nodes[tId + num].surface_index)]);

					num++;
				}
			}
			nodes_new[counter[tId]] = nodes[tId];
			nodes_object[counter[tId]] = pobject;
			nodes_normal[counter[tId]] = pnormal;

			dist = (pos - nodes_object[counter[tId]]).norm();
			Real sign = (pos - nodes_object[counter[tId]]).dot(nodes_normal[counter[tId]]) < Real(0) ? Real(-1) : Real(1);
			nodes_sdf[counter[tId]] = sign * dist;

			x_index[counter[tId]] = IndexNode((gnz*nx_*ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx*ny_*nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny*nz_*nx_ + gnx * nz_ + gnz), counter[tId]);

		}
	}

	//根据现有的最底层网格进行矩阵求解来计算新网格的投影点和法向（根据法线垂直于表面来求解）
	template <typename Real, typename Coord>
	__global__ void SO_FetchAndInterpolateReconstructedGrids2(
		DArray<PositionNode> nodes_new,
		DArray<Real> nodes_sdf,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<int> counter,
		DArray<PositionNode> nodes,
		DArray<VoxelOctreeNode<Coord>> nodes_old,
		DArray<Coord> object,
		DArray<Coord> normal,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= counter.size()) return;

		if (tId == 0 || nodes[tId].position_index != nodes[tId - 1].position_index)
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);
			Coord pos_old(nodes_old[nodes[tId].surface_index].position());

			int multiple = 4;

			Real dist = (pos - pos_old).norm();
			Real weight_sum(1.0f), weight(1.0f);
			kernel3(weight_sum, dist / (multiple*dx_));

			int num = 1;
			while (nodes[tId + num].position_index == nodes[tId].position_index)
			{
				pos_old = nodes_old[nodes[tId + num].surface_index].position();
				dist = (pos - pos_old).norm();
				kernel3(weight, dist / (multiple*dx_));

				weight_sum += weight;
				num++;
			}

			Coord pobject(object[nodes[tId].surface_index]);
			Coord pnormal(normal[nodes[tId].surface_index]);

			if (weight_sum > 10 * REAL_EPSILON)
			{
				pos_old = nodes_old[nodes[tId].surface_index].position();
				dist = (pos - pos_old).norm();
				kernel3(weight, dist / (multiple*dx_));
				weight = weight / weight_sum;
				pnormal = pnormal * weight;

				Vec3d b(pnormal[0], pnormal[1], pnormal[2]);
				b *= (weight*(pobject.dot(pnormal)));
				//b *= (pobject.dot(pnormal));

				Mat3d M(pnormal[0] * pnormal[0], pnormal[0] * pnormal[1], pnormal[0] * pnormal[2],
					pnormal[1] * pnormal[0], pnormal[1] * pnormal[1], pnormal[1] * pnormal[2],
					pnormal[2] * pnormal[0], pnormal[2] * pnormal[1], pnormal[2] * pnormal[2]);
				M *= weight;

				num = 1;
				while (nodes[tId + num].position_index == nodes[tId].position_index)
				{
					pos_old = nodes_old[nodes[tId + num].surface_index].position();
					dist = (pos - pos_old).norm();
					kernel3(weight, dist / (multiple*dx_));
					weight = weight / weight_sum;

					Coord pobj(object[(nodes[tId + num].surface_index)]);
					Coord pnor(normal[(nodes[tId + num].surface_index)]);
					pnormal += (weight*pnor);

					Vec3d b_i(pnor[0], pnor[1], pnor[2]);
					b += (weight*b_i*(pobj.dot(pnor)));

					M(0, 0) += weight * pnor[0] * pnor[0];	M(0, 1) += weight * pnor[0] * pnor[1];	M(0, 2) += weight * pnor[0] * pnor[2];
					M(1, 0) += weight * pnor[1] * pnor[0];	M(1, 1) += weight * pnor[1] * pnor[1];	M(1, 2) += weight * pnor[1] * pnor[2];
					M(2, 0) += weight * pnor[2] * pnor[0];	M(2, 1) += weight * pnor[2] * pnor[1];	M(2, 2) += weight * pnor[2] * pnor[2];

					num++;
				}
				Mat3d M_inv = M.inverse();
				Vec3d x = M_inv * b;
				pobject = Coord(x[0], x[1], x[2]);
			}

			nodes_new[counter[tId]] = nodes[tId];
			nodes_object[counter[tId]] = pobject;
			nodes_normal[counter[tId]] = pnormal;

			dist = (pos - nodes_object[counter[tId]]).norm();
			Real sign = (pos - nodes_object[counter[tId]]).dot(nodes_normal[counter[tId]]) < Real(0) ? Real(-1) : Real(1);
			nodes_sdf[counter[tId]] = sign * dist;

			x_index[counter[tId]] = IndexNode((gnz*nx_*ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx*ny_*nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny*nz_*nx_ + gnx * nz_ + gnz), counter[tId]);
		}
	}

	template <typename Real, typename Coord>
	GPU_FUNC void SO_UpdateReconstructionGrid(
		DArray<Real>& nodes_sdf,
		DArray<Coord>& nodes_object,
		DArray<Coord>& nodes_normal,
		Coord ppos,
		int id,
		int nid)
	{
		Real dist = (ppos - nodes_object[nid]).norm();
		if (abs(dist) < abs(nodes_sdf[id]))
		{
			nodes_object[id] = nodes_object[nid];
			nodes_normal[id] = nodes_normal[nid];

			Real sign = (ppos - nodes_object[id]).dot(nodes_normal[id]) < Real(0) ? Real(-1) : Real(1);

			nodes_sdf[id] = sign * dist;
		}
	}

	template <typename Real, typename Coord>
	__global__ void SO_FIMUpdateReconstructionGrids(
		DArray<PositionNode> nodes,
		DArray<Real> nodes_sdf,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		OcIndex gnx, gny, gnz;
		RecoverFromMortonCode((OcKey)(nodes[tId].position_index), gnx, gny, gnz);
		Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

		int xnx = (x_index[tId].xyz_index) % nx_;
		if ((xnx != 0 && tId > 0) && (x_index[tId - 1].xyz_index == (x_index[tId].xyz_index - 1)))
			SO_UpdateReconstructionGrid(nodes_sdf, nodes_object, nodes_normal, pos, (x_index[tId].node_index), (x_index[tId - 1].node_index));
		if ((xnx != (nx_ - 1) && tId < (nodes.size() - 1)) && (x_index[tId + 1].xyz_index == (x_index[tId].xyz_index + 1)))
			SO_UpdateReconstructionGrid(nodes_sdf, nodes_object, nodes_normal, pos, (x_index[tId].node_index), (x_index[tId + 1].node_index));

		int yny = (y_index[tId].xyz_index) % ny_;
		if ((yny != 0 && tId > 0) && (y_index[tId - 1].xyz_index == (y_index[tId].xyz_index - 1)))
			SO_UpdateReconstructionGrid(nodes_sdf, nodes_object, nodes_normal, pos, (y_index[tId].node_index), (y_index[tId - 1].node_index));
		if ((yny != (ny_ - 1) && tId < (nodes.size() - 1)) && (y_index[tId + 1].xyz_index == (y_index[tId].xyz_index + 1)))
			SO_UpdateReconstructionGrid(nodes_sdf, nodes_object, nodes_normal, pos, (y_index[tId].node_index), (y_index[tId + 1].node_index));

		int znz = (z_index[tId].xyz_index) % nz_;
		if ((znz != 0 && tId > 0) && (z_index[tId - 1].xyz_index == (z_index[tId].xyz_index - 1)))
			SO_UpdateReconstructionGrid(nodes_sdf, nodes_object, nodes_normal, pos, (z_index[tId].node_index), (z_index[tId - 1].node_index));
		if ((znz != (nz_ - 1) && tId < (nodes.size() - 1)) && (z_index[tId + 1].xyz_index == (z_index[tId].xyz_index + 1)))
			SO_UpdateReconstructionGrid(nodes_sdf, nodes_object, nodes_normal, pos, (z_index[tId].node_index), (z_index[tId + 1].node_index));
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::finestLevelReconstruction(
		DArray<PositionNode>& recon_node,
		DArray<Real>& recon_sdf,
		DArray<Coord>& recon_object,
		DArray<Coord>& recon_normal,
		DArray<VoxelOctreeNode<Coord>>& grid,
		DArray<Coord>& grid_object,
		DArray<Coord>& grid_normal,
		Coord m_origin,
		Real m_dx,
		int m_nx,
		int m_ny,
		int m_nz,
		Real m_dx_old,
		int level_0,
		int& level_0_recon)
	{
		DArray<int> count;
		count.resize(level_0);
		//数重建网格的数目(with repeat)
		cuExecute(count.size(),
			SO_CountReconstructedGrids,
			count,
			grid,
			m_origin,
			m_dx,
			m_dx_old,
			m_nx,
			m_ny,
			m_nz);

		int grid_num = thrust::reduce(thrust::device, count.begin(), count.begin() + count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, count.begin(), count.begin() + count.size(), count.begin());
		//std::printf("the reconstructed grids(with repeat) is: %d %d \n", sdfOctree_a->getLevel0(), grid_num);

		DArray<PositionNode> grid_buf;
		grid_buf.resize(grid_num);
		//取出重建的网格(with repeat)
		cuExecute(count.size(),
			SO_FetchReconstructedGrids,
			grid_buf,
			count,
			grid,
			m_origin,
			m_dx,
			m_dx_old,
			m_nx,
			m_ny,
			m_nz);

		thrust::sort(thrust::device, grid_buf.begin(), grid_buf.begin() + grid_buf.size(), PositionCmp());

		count.resize(grid_num);
		count.reset();
		//数重建网格不重复的网格
		cuExecute(grid_num,
			SO_CountNoRepeatedReconstructedGrids,
			count,
			grid_buf);

		level_0_recon = thrust::reduce(thrust::device, count.begin(), count.begin() + count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, count.begin(), count.begin() + count.size(), count.begin());

		DArray<IndexNode> xIndex, yIndex, zIndex;
		xIndex.resize(level_0_recon);
		yIndex.resize(level_0_recon);
		zIndex.resize(level_0_recon);
		recon_node.resize(level_0_recon);
		recon_sdf.resize(level_0_recon);
		recon_object.resize(level_0_recon);
		recon_normal.resize(level_0_recon);
		//取出重建网格不重复的网格
		//cuExecute(grid_num,
		//	SO_FetchNoRepeatedReconstructedGrids,
		//	recon_node,
		//	recon_sdf,
		//	recon_object,
		//	recon_normal,
		//	count,
		//	grid_buf,
		//	grid,
		//	grid_object,
		//	grid_normal,
		//	m_origin,
		//	m_dx);
		cuExecute(grid_num,
			SO_FetchAndInterpolateReconstructedGrids,
			recon_node,
			recon_sdf,
			recon_object,
			recon_normal,
			xIndex,
			yIndex,
			zIndex,
			count,
			grid_buf,
			grid,
			grid_object,
			grid_normal,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);
		cuExecute(grid_num,
			SO_FIMUpdateReconstructionGrids,
			recon_node,
			recon_sdf,
			recon_object,
			recon_normal,
			xIndex,
			yIndex,
			zIndex,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);

		count.clear();
		grid_buf.clear();
		xIndex.clear();
		yIndex.clear();
		zIndex.clear();
	}

	template <typename Coord>
	__global__ void SO_NodesInitialReconstruction(
		DArray<PositionNode> nodes,
		DArray<Coord> pos,
		DArray<PositionNode> nodes_a,
		DArray<VoxelOctreeNode<Coord>> nodes_b,
		int off_bx,
		int off_by,
		int off_bz,
		int level0_a,
		Coord origin_,
		Real dx_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= nodes.size()) return;

		if (tId < level0_a)
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(nodes_a[tId].position_index), gnx, gny, gnz);
			Coord pos0(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			nodes[tId] = PositionNode(tId, nodes_a[tId].position_index);
			pos[tId] = pos0;
		}
		else
		{
			OcIndex gnx0, gny0, gnz0, gnx1, gny1, gnz1;
			OcKey old_key = nodes_b[tId - level0_a].key();
			RecoverFromMortonCode(old_key, gnx0, gny0, gnz0);

			gnx1 = gnx0 + off_bx;
			gny1 = gny0 + off_by;
			gnz1 = gnz0 + off_bz;
			OcKey new_key = CalculateMortonCode(gnx1, gny1, gnz1);

			nodes[tId] = PositionNode(tId, new_key);
			pos[tId] = nodes_b[tId - level0_a].position();
		}
	}

	//注意counter中第i+1个元素存的是0-i元素的和
	template <typename Real, typename Coord>
	__global__ void SO_FetchNonRepeatedGridsReconstruction(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal,
		DArray<IndexNode> x_index,
		DArray<IndexNode> y_index,
		DArray<IndexNode> z_index,
		DArray<PositionNode> all_nodes,
		DArray<int> counter,
		DArray<Real> nodes_value_a,
		DArray<Coord> nodes_object_a,
		DArray<Coord> nodes_normal_a,
		DArray<VoxelOctreeNode<Coord>> nodes_b,
		DArray<Real> nodes_value_b,
		DArray<Coord> nodes_object_b,
		DArray<Coord> nodes_normal_b,
		int level0_a,
		int op,
		Coord origin_,
		Real dx_,
		int nx_,
		int ny_,
		int nz_)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= counter.size()) return;

		if (tId < (counter.size() - 1) && counter[tId] != counter[tId + 1])
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(all_nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			VoxelOctreeNode<Coord> mc((Level)0, gnx, gny, gnz, pos);
			mc.setValueLocation(counter[tId]);

			if (all_nodes[tId].surface_index < level0_a)
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_a[all_nodes[tId].surface_index];
				nodes_object[counter[tId]] = nodes_object_a[all_nodes[tId].surface_index];
				if (op == 3)
					nodes_normal[counter[tId]] = -nodes_normal_a[all_nodes[tId].surface_index];
				else
					nodes_normal[counter[tId]] = nodes_normal_a[all_nodes[tId].surface_index];
			}
			else
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_b[all_nodes[tId].surface_index - level0_a];
				nodes_object[counter[tId]] = nodes_object_b[all_nodes[tId].surface_index - level0_a];
				if (op == 2)
					nodes_normal[counter[tId]] = -nodes_normal_b[all_nodes[tId].surface_index - level0_a];
				else
					nodes_normal[counter[tId]] = nodes_normal_b[all_nodes[tId].surface_index - level0_a];
			}

			x_index[counter[tId]] = IndexNode((gnz*nx_*ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx*ny_*nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny*nz_*nx_ + gnx * nz_ + gnz), counter[tId]);
		}
		else if (tId == (counter.size() - 1) && (counter[tId] < nodes.size()))
		{
			OcIndex gnx, gny, gnz;
			RecoverFromMortonCode((OcKey)(all_nodes[tId].position_index), gnx, gny, gnz);
			Coord pos(origin_[0] + (gnx + 0.5) * dx_, origin_[1] + (gny + 0.5) * dx_, origin_[2] + (gnz + 0.5) * dx_);

			VoxelOctreeNode<Coord> mc((Level)0, gnx, gny, gnz, pos);
			mc.setValueLocation(counter[tId]);

			if (all_nodes[tId].surface_index < level0_a)
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_a[all_nodes[tId].surface_index];
				nodes_object[counter[tId]] = nodes_object_a[all_nodes[tId].surface_index];
				if (op == 3)
					nodes_normal[counter[tId]] = -nodes_normal_a[all_nodes[tId].surface_index];
				else
					nodes_normal[counter[tId]] = nodes_normal_a[all_nodes[tId].surface_index];
			}
			else
			{
				nodes[counter[tId]] = mc;
				nodes_value[counter[tId]] = nodes_value_b[all_nodes[tId].surface_index - level0_a];
				nodes_object[counter[tId]] = nodes_object_b[all_nodes[tId].surface_index - level0_a];
				if (op == 2)
					nodes_normal[counter[tId]] = -nodes_normal_b[all_nodes[tId].surface_index - level0_a];
				else
					nodes_normal[counter[tId]] = nodes_normal_b[all_nodes[tId].surface_index - level0_a];
			}

			x_index[counter[tId]] = IndexNode((gnz*nx_*ny_ + gny * nx_ + gnx), counter[tId]);
			y_index[counter[tId]] = IndexNode((gnx*ny_*nz_ + gnz * ny_ + gny), counter[tId]);
			z_index[counter[tId]] = IndexNode((gny*nz_*nx_ + gnx * nz_ + gnz), counter[tId]);
		}
	}


	template <typename Real, typename Coord>
	__global__ void SO_FIMUpdateGrids(
		DArray<VoxelOctreeNode<Coord>> nodes,
		DArray<Real> nodes_value,
		DArray<Coord> nodes_object,
		DArray<Coord> nodes_normal)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= nodes.size()) return;

		for (int i = 0; i < 6; i++)
		{
			if (nodes[tId].m_neighbor[i] != EMPTY)
			{
				int nb_id = nodes[tId].m_neighbor[i];
				Real sign = (nodes[tId].position() - nodes_object[nb_id]).dot(nodes_normal[nb_id]) < Real(0) ? Real(-1) : Real(1);
				Real dist = sign * (nodes[tId].position() - nodes_object[nb_id]).norm();

				if (abs(dist) < abs(nodes_value[tId]))
				{
					nodes_value[tId] = dist;
					nodes_object[tId] = nodes_object[nb_id];
					nodes_normal[tId] = nodes_normal[nb_id];
				}
			}
		}
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::finestLevelReconstBoolean(
		DArray<VoxelOctreeNode<Coord>>& grid0,
		DArray<Real>& grid0_value,
		DArray<Coord>& grid0_object,
		DArray<Coord>& grid0_normal,
		DArray<PositionNode>& recon_node,
		DArray<Real>& recon_sdf,
		DArray<Coord>& recon_object,
		DArray<Coord>& recon_normal,
		DArray<VoxelOctreeNode<Coord>>& sdfOctreeNode_2,
		DArray<Real>& sdfValue_2,
		DArray<Coord>& object_2,
		DArray<Coord>& normal_2,
		std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_1,
		std::shared_ptr<VoxelOctree<TDataType>> sdfOctree_2,
		int level0_1,
		int level0_2,
		int& m_level0,
		int offset_nx,
		int offset_ny,
		int offset_nz,
		Coord m_origin,
		Real m_dx,
		int m_nx,
		int m_ny,
		int m_nz,
		int boolean)//boolean=0:union, boolean=1:intersection, boolean=2:subtraction, boolean=3:subtraction with model b reconstruction
	{
		int level0_num = level0_1 + level0_2;
		DArray<PositionNode> grid_buf;
		grid_buf.resize(level0_num);
		DArray<Coord> grid_pos;
		grid_pos.resize(level0_num);
		//重建网格中不重复的网格与level0的网格放在一起
		cuExecute(level0_num,
			SO_NodesInitialReconstruction,
			grid_buf,
			grid_pos,
			recon_node,
			sdfOctreeNode_2,
			offset_nx,
			offset_ny,
			offset_nz,
			level0_1,
			m_origin,
			m_dx);

		thrust::sort(thrust::device, grid_buf.begin(), grid_buf.begin() + grid_buf.size(), PositionCmp());

		DArray<Real> pos_value_1, pos_value_2;
		DArray<Coord> pos_normal_1, pos_normal_2;
		sdfOctree_1->getSignDistanceMLS(grid_pos, pos_value_1, pos_normal_1);
		sdfOctree_2->getSignDistanceMLS(grid_pos, pos_value_2, pos_normal_2);
		pos_normal_1.clear();
		pos_normal_2.clear();

		DArray<int> count;
		count.resize(level0_num);
		count.reset();
		//数不重复的grid
		if (boolean == 0)
		{
			cuExecute(count.size(),
				SO_CountNonRepeatedUnion,
				count,
				grid_buf,
				recon_sdf,
				sdfValue_2,
				pos_value_1,
				pos_value_2,
				level0_1);
		}
		else if (boolean == 1)
		{
			cuExecute(count.size(),
				SO_CountNonRepeatedIntersection,
				count,
				grid_buf,
				recon_sdf,
				sdfValue_2,
				pos_value_1,
				pos_value_2,
				level0_1);
		}
		else if (boolean == 2)
		{
			cuExecute(count.size(),
				SO_CountNonRepeatedSubtractionA,
				count,
				grid_buf,
				recon_sdf,
				sdfValue_2,
				pos_value_1,
				pos_value_2,
				level0_1);
		}
		else if (boolean == 3)
		{
			cuExecute(count.size(),
				SO_CountNonRepeatedSubtractionB,
				count,
				grid_buf,
				recon_sdf,
				sdfValue_2,
				pos_value_1,
				pos_value_2,
				level0_1);
		}
		int grid0_num = thrust::reduce(thrust::device, count.begin(), count.begin() + count.size(), (int)0, thrust::plus<int>());
		thrust::exclusive_scan(thrust::device, count.begin(), count.begin() + count.size(), count.begin());

		if (grid0_num == 0) { return; }

		m_level0 = grid0_num;

		DArray<IndexNode> xIndex, yIndex, zIndex;
		xIndex.resize(grid0_num);
		yIndex.resize(grid0_num);
		zIndex.resize(grid0_num);
		grid0.resize(grid0_num);
		grid0_value.resize(grid0_num);
		grid0_object.resize(grid0_num);
		grid0_normal.resize(grid0_num);
		//去重
		cuExecute(count.size(),
			SO_FetchNonRepeatedGridsReconstruction,
			grid0,
			grid0_value,
			grid0_object,
			grid0_normal,
			xIndex,
			yIndex,
			zIndex,
			grid_buf,
			count,
			recon_sdf,
			recon_object,
			recon_normal,
			sdfOctreeNode_2,
			sdfValue_2,
			object_2,
			normal_2,
			level0_1,
			boolean,
			m_origin,
			m_dx,
			m_nx,
			m_ny,
			m_nz);

		thrust::sort(thrust::device, xIndex.begin(), xIndex.begin() + xIndex.size(), IndexCmp());
		thrust::sort(thrust::device, yIndex.begin(), yIndex.begin() + yIndex.size(), IndexCmp());
		thrust::sort(thrust::device, zIndex.begin(), zIndex.begin() + zIndex.size(), IndexCmp());

		cuExecute(grid0_num,
			SO_UpdateBottomNeighbors,
			grid0,
			xIndex,
			yIndex,
			zIndex,
			m_nx,
			m_ny,
			m_nz);

		for (int i = 0; i < 3; i++)
		{
			cuExecute(grid0_num,
				SO_FIMUpdateGrids,
				grid0,
				grid0_value,
				grid0_object,
				grid0_normal);
		}

		xIndex.clear();
		yIndex.clear();
		zIndex.clear();
		count.clear();
		grid_buf.clear();
		grid_pos.clear();
		pos_value_1.clear();
		pos_value_2.clear();
	}

	template <typename Real>
	__global__ void SO_IntersectionSet(
		DArray<Real> inter_value,
		DArray<int> inter_index,
		DArray<Real> inter_value_a,
		DArray<Real> inter_value_b)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= inter_index.size()) return;

		if (inter_value_a[tId] < 0 && inter_value_b[tId] < 0)
			inter_value[inter_index[tId]] = -std::abs(inter_value[inter_index[tId]]);
		else
			inter_value[inter_index[tId]] = std::abs(inter_value[inter_index[tId]]);
	}

	template <typename Real>
	__global__ void SO_UnionSet(
		DArray<Real> inter_value,
		DArray<int> inter_index,
		DArray<Real> inter_value_a,
		DArray<Real> inter_value_b)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= inter_index.size()) return;

		if (inter_value_a[tId] < 0 || inter_value_b[tId] < 0)
			inter_value[inter_index[tId]] = -std::abs(inter_value[inter_index[tId]]);
		else
			inter_value[inter_index[tId]] = std::abs(inter_value[inter_index[tId]]);
	}

	template <typename Real>
	__global__ void SO_SubtractionSetA(
		DArray<Real> inter_value,
		DArray<int> inter_index,
		DArray<Real> inter_value_a,
		DArray<Real> inter_value_b)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);

		if (tId >= inter_index.size()) return;

		if (inter_value_a[tId] <= 0 && inter_value_b[tId] >= 0)
			inter_value[inter_index[tId]] = -std::abs(inter_value[inter_index[tId]]);
		else
			inter_value[inter_index[tId]] = std::abs(inter_value[inter_index[tId]]);
	}

	template<typename TDataType>
	void VolumeHelper<TDataType>::updateBooleanSigned(
		DArray<Real>& leaf_value,
		DArray<int>& leaf_index,
		DArray<Real>& leaf_value_a,
		DArray<Real>& leaf_value_b,
		int boolean) //boolean = 0:union, boolean = 1 : intersection, boolean = 2 : subtraction
	{
		if (boolean == 0)
		{
			cuExecute(leaf_index.size(),
				SO_UnionSet,
				leaf_value,
				leaf_index,
				leaf_value_a,
				leaf_value_b);

		}
		else if (boolean == 1)
		{
			cuExecute(leaf_index.size(),
				SO_IntersectionSet,
				leaf_value,
				leaf_index,
				leaf_value_a,
				leaf_value_b);
		}
		else if (boolean == 2)
		{
			cuExecute(leaf_index.size(),
				SO_SubtractionSetA,
				leaf_value,
				leaf_index,
				leaf_value_a,
				leaf_value_b);
		}
	}
	DEFINE_CLASS(VolumeHelper);
}