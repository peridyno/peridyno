#include <fstream>
#include "DistanceField3D.h"
#include "Vector.h"
#include "DataTypes.h"

namespace dyno{

	template <typename Coord>
	__device__  float DistanceToPlane(const Coord &p, const Coord &o, const Coord &n) {
		return fabs((p - o, n).length());
	}

	template <typename Coord>
	__device__  Real DistanceToSegment(Coord& pos, Coord& lo, Coord& hi)
	{
		typedef typename Coord::VarType Real;
		Coord seg = hi - lo;
		Coord edge1 = pos - lo;
		Coord edge2 = pos - hi;
		if (edge1.dot(seg) < 0.0f)
		{
			return edge1.norm();
		}
		if (edge2.dot(-seg) < 0.0f)
		{
			return edge2.norm();
		}
		Real length1 = edge1.dot(edge1);
		seg.normalize();
		Real length2 = edge1.dot(seg);
		return std::sqrt(length1 - length2*length2);
	}

	template <typename Coord>
	__device__  Real DistanceToSqure(Coord& pos, Coord& lo, Coord& hi, int axis)
	{
		typedef typename Coord::VarType Real;
		Coord n;
		Coord corner1, corner2, corner3, corner4;
		Coord loCorner, hiCorner, p;
		switch (axis)
		{
		case 0:
			corner1 = Coord(lo[0], lo[1], lo[2]);
			corner2 = Coord(lo[0], hi[1], lo[2]);
			corner3 = Coord(lo[0], hi[1], hi[2]);
			corner4 = Coord(lo[0], lo[1], hi[2]);
			n = Coord(1.0, 0.0, 0.0);

			loCorner = Coord(lo[1], lo[2], 0.0);
			hiCorner = Coord(hi[1], hi[2], 0.0);
			p = Coord(pos[1], pos[2], 0.0f);
			break;
		case 1:
			corner1 = Coord(lo[0], lo[1], lo[2]);
			corner2 = Coord(lo[0], lo[1], hi[2]);
			corner3 = Coord(hi[0], lo[1], hi[2]);
			corner4 = Coord(hi[0], lo[1], lo[2]);
			n = Coord(0.0f, 1.0f, 0.0f);

			loCorner = Coord(lo[0], lo[2], 0.0f);
			hiCorner = Coord(hi[0], hi[2], 0.0f);
			p = Coord(pos[0], pos[2], 0.0f);
			break;
		case 2:
			corner1 = Coord(lo[0], lo[1], lo[2]);
			corner2 = Coord(hi[0], lo[1], lo[2]);
			corner3 = Coord(hi[0], hi[1], lo[2]);
			corner4 = Coord(lo[0], hi[1], lo[2]);
			n = Coord(0.0f, 0.0f, 1.0f);

			loCorner = Coord(lo[0], lo[1], 0.0);
			hiCorner = Coord(hi[0], hi[1], 0.0);
			p = Coord(pos[0], pos[1], 0.0f);
			break;
		}

		Real dist1 = DistanceToSegment(pos, corner1, corner2);
		Real dist2 = DistanceToSegment(pos, corner2, corner3);
		Real dist3 = DistanceToSegment(pos, corner3, corner4);
		Real dist4 = DistanceToSegment(pos, corner4, corner1);
		Real dist5 = abs(n.dot(pos - corner1));
		if (p[0] < hiCorner[0] && p[0] > loCorner[0] && p[1] < hiCorner[1] && p[1] > loCorner[1])
			return dist5;
		else
			return min(min(dist1, dist2), min(dist3, dist4));
	}

	template <typename Coord>
	__device__  Real DistanceToBox(Coord& pos, Coord& lo, Coord& hi)
	{
		typedef typename Coord::VarType Real;
		Coord corner0(lo[0], lo[1], lo[2]);
		Coord corner1(hi[0], lo[1], lo[2]);
		Coord corner2(hi[0], hi[1], lo[2]);
		Coord corner3(lo[0], hi[1], lo[2]);
		Coord corner4(lo[0], lo[1], hi[2]);
		Coord corner5(hi[0], lo[1], hi[2]);
		Coord corner6(hi[0], hi[1], hi[2]);
		Coord corner7(lo[0], hi[1], hi[2]);
		Real dist0 = (pos - corner0).norm();
		Real dist1 = (pos - corner1).norm();
		Real dist2 = (pos - corner2).norm();
		Real dist3 = (pos - corner3).norm();
		Real dist4 = (pos - corner4).norm();
		Real dist5 = (pos - corner5).norm();
		Real dist6 = (pos - corner6).norm();
		Real dist7 = (pos - corner7).norm();
		if (pos[0] < hi[0] && pos[0] > lo[0] && pos[1] < hi[1] && pos[1] > lo[1] && pos[2] < hi[2] && pos[2] > lo[2])
		{
			Real distx = min(abs(pos[0] - hi[0]), abs(pos[0] - lo[0]));
			Real disty = min(abs(pos[1] - hi[1]), abs(pos[1] - lo[1]));
			Real distz = min(abs(pos[2] - hi[2]), abs(pos[2] - lo[2]));
			Real mindist = min(distx, disty);
			mindist = min(mindist, distz);
			return mindist;
		}
		else
		{
			Real distx1 = DistanceToSqure(pos, corner0, corner7, 0);
			Real distx2 = DistanceToSqure(pos, corner1, corner6, 0);
			Real disty1 = DistanceToSqure(pos, corner0, corner5, 1);
			Real disty2 = DistanceToSqure(pos, corner3, corner6, 1);
			Real distz1 = DistanceToSqure(pos, corner0, corner2, 2);
			Real distz2 = DistanceToSqure(pos, corner4, corner6, 2);
			return -min(min(min(distx1, distx2), min(disty1, disty2)), min(distz1, distz2));
		}
	}

	template <typename Real, typename Coord>
	__device__  Real DistanceToCylinder(Coord& pos, Coord& center, Real radius, Real height, int axis)
	{
		Real distR;
		Real distH;
		switch (axis)
		{
		case 0:
			distH = abs(pos[0] - center[0]);
			distR = Coord(0.0, pos[1] - center[1], pos[2] - center[2]).norm();
			break;
		case 1:
			distH = abs(pos[1] - center[1]);
			distR = Coord(pos[0] - center[0], 0.0, pos[2] - center[2]).norm();
			break;
		case 2:
			distH = abs(pos[2] - center[2]);
			distR = Coord(pos[0] - center[0], pos[1] - center[1], 0.0).norm();
			break;
		}

		Real halfH = height / 2.0f;
		if (distH <= halfH && distR <= radius)
		{
			return -min(halfH - distH, radius - distR);
		}
		else if (distH > halfH && distR <= radius)
		{
			return distH - halfH;
		}
		else if (distH <= halfH && distR > radius)
		{
			return distR - radius;
		}
		else
		{
// 			Real l1 = distR - radius;
// 			Real l2 = distH - halfH;
//			return sqrt(l1*l1 + l2*l2);
			return Vector<Real, 2>(distR - radius, distH - halfH).norm();
		}


	}

	template <typename Real, typename Coord>
	__device__  Real DistanceToSphere(Coord& pos, Coord& center, Real radius)
	{
		return (pos - center).length() - radius;
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::DistanceField3D()
	{
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::DistanceField3D(std::string filename)
	{
		loadSDF(filename);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::setSpace(const Coord p0, const Coord p1, int nbx, int nby, int nbz)
	{
		m_left = p0;

		m_h = (p1 - p0)*Coord(1.0 / Real(nbx+1), 1.0 / Real(nby+1), 1.0 / Real(nbz+1));

		m_distance.resize(nbx+1, nby+1, nbz+1);
	}

	template<typename TDataType>
	DistanceField3D<TDataType>::~DistanceField3D()
	{
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::translate(const Coord &t) {
		m_left += t;
	}

	template <typename Real>
	__global__ void K_Scale(DArray3D<Real> distance, float s)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		distance(i, j, k) = s*distance(i, j, k);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::scale(const Real s) {
		m_left[0] *= s;
		m_left[1] *= s;
		m_left[2] *= s;
		m_h[0] *= s;
		m_h[1] *= s;
		m_h[2] *= s;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(m_distance.nx(), m_distance.ny(), m_distance.nz()), blockSize);

		K_Scale << <gridDims, blockSize >> >(m_distance, s);
	}

	template<typename Real>
	__global__ void K_Invert(DArray3D<Real> distance)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		distance(i, j, k) = -distance(i, j, k);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::invertSDF()
	{
		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(m_distance.nx(), m_distance.ny(), m_distance.nz()), blockSize);

		K_Invert << <gridDims, blockSize >> >(m_distance);
	}

	template <typename Real, typename Coord>
	__global__ void K_DistanceFieldToBox(DArray3D<Real> distance, Coord start, Coord h, Coord lo, Coord hi, bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		int sign = inverted ? 1.0f : -1.0f;
		Coord p = start + Coord(i, j, k)*h;

		distance(i, j, k) = sign*DistanceToBox(p, lo, hi);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadBox(Coord& lo, Coord& hi, bool inverted)
	{
		m_bInverted = inverted;

		dim3 blockSize = make_uint3(4, 4, 4);
		dim3 gridDims = cudaGridSize3D(make_uint3(m_distance.nx(), m_distance.ny(), m_distance.nz()), blockSize);

		K_DistanceFieldToBox << <gridDims, blockSize >> >(m_distance, m_left, m_h, lo, hi, inverted);
	}

	template <typename Real, typename Coord>
	__global__ void K_DistanceFieldToCylinder(DArray3D<Real> distance, Coord start, Coord h, Coord center, Real radius, Real height, int axis, bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		int sign = inverted ? -1.0f : 1.0f;

		Coord p = start + Coord(i, j, k)*h;

		distance(i, j, k) = sign*DistanceToCylinder(p, center, radius, height, axis);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadCylinder(Coord& center, Real radius, Real height, int axis, bool inverted)
	{
		m_bInverted = inverted;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(m_distance.nx(), m_distance.ny(), m_distance.nz()), blockSize);

		K_DistanceFieldToCylinder << <gridDims, blockSize >> >(m_distance, m_left, m_h, center, radius, height, axis, inverted);
	}

	template <typename Real, typename Coord>
	__global__ void K_DistanceFieldToSphere(DArray3D<Real> distance, Coord start, Coord h, Coord center, Real radius, bool inverted)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		int j = threadIdx.y + (blockIdx.y * blockDim.y);
		int k = threadIdx.z + (blockIdx.z * blockDim.z);

		if (i >= distance.nx()) return;
		if (j >= distance.ny()) return;
		if (k >= distance.nz()) return;

		int sign = inverted ? -1.0f : 1.0f;

		Coord p = start + Coord(i, j, k)*h;

		Coord dir = p - center;

		distance(i, j, k) = sign*(dir.norm()-radius);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadSphere(Coord& center, Real radius, bool inverted)
	{
		m_bInverted = inverted;

		dim3 blockSize = make_uint3(8, 8, 8);
		dim3 gridDims = cudaGridSize3D(make_uint3(m_distance.nx(), m_distance.ny(), m_distance.nz()), blockSize);

		K_DistanceFieldToSphere << <gridDims, blockSize >> >(m_distance, m_left, m_h, center, radius, inverted);
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::loadSDF(std::string filename, bool inverted)
	{
		std::ifstream input(filename.c_str(), std::ios::in);
		if (!input.is_open())
		{
			std::cout << "Reading file " << filename << " error!" << std::endl;
			exit(0);
		}

		int nbx, nby, nbz;
		int xx, yy, zz;

		input >> xx;
		input >> yy;
		input >> zz;

		input >> m_left[0];
		input >> m_left[1];
		input >> m_left[2];

		Real t_h;
		input >> t_h;

		std::cout << "SDF: " << xx << ", " << yy << ", " << zz << std::endl;
		std::cout << "SDF: " << m_left[0] << ", " << m_left[1] << ", " << m_left[2] << std::endl;
		std::cout << "SDF: " << m_left[0] + t_h*xx << ", " << m_left[1] + t_h*yy << ", " << m_left[2] + t_h*zz << std::endl;

		nbx = xx;
		nby = yy;
		nbz = zz;
		m_h[0] = t_h;
		m_h[1] = t_h;
		m_h[2] = t_h;

		CArray3D<Real> distances(nbx, nby, nbz);
		for (int k = 0; k < zz; k++) {
			for (int j = 0; j < yy; j++) {
				for (int i = 0; i < xx; i++) {
					float dist;
					input >> dist;
					distances(i, j, k) = dist;
				}
			}
		}
		input.close();

		m_distance.resize(nbx, nby, nbz);
		m_distance.assign(distances);
		//cuSafeCall(cudaMemcpy(m_distance.begin(), distances, (nbx)*(nby)*(nbz) * sizeof(Real), cudaMemcpyHostToDevice));


		m_bInverted = inverted;
		if (inverted)
		{
			invertSDF();
		}

		distances.clear();

		std::cout << "read data successful" << std::endl;
	}

	template<typename TDataType>
	void DistanceField3D<TDataType>::release()
	{
		m_distance.clear();
	}

	template class DistanceField3D<DataType3f>;
	template class DistanceField3D<DataType3d>;
}