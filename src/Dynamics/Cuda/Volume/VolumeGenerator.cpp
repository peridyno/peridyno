#include "VolumeGenerator.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

namespace dyno
{
	IMPLEMENT_TCLASS(VolumeGenerator, TDataType)

	template<typename TDataType>
	VolumeGenerator<TDataType>::VolumeGenerator()
		//: Volume()
	{
		this->inPadding()->setValue(10);
		this->inSpacing()->setValue(0.05);
	}

	template<typename TDataType>
	VolumeGenerator<TDataType>::~VolumeGenerator()
	{
	}

	template<typename TDataType>
	void VolumeGenerator<TDataType>::resetStates()
	{
		if (this->inTriangleSet()->isEmpty() == false) {
			this->loadClosedSurface();

			if (this->outGenSDF()->isEmpty()) {
				this->outGenSDF()->allocate();
			}
			DistanceField3D<TDataType>& sdf = this->outGenSDF()->getDataPtr()->getSDF();

			Coord p0(origin.x, origin.y, origin.z);
			Coord p1(maxPoint.x, maxPoint.y, maxPoint.z);

			sdf.setSpace(p0, p1, ni - 1, nj - 1, nk - 1);
			sdf.setDistance(phi);
		}

		printf("VolumeGenerator ok \n");
	}

	template<typename TDataType>
	void VolumeGenerator<TDataType>::loadClosedSurface()
	{
		Vec3f min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
		Vec3f max_box(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

		DArray<Coord> mPoints = inTriangleSet()->getDataPtr()->getPoints();
		DArray<Triangle> mTriangles = inTriangleSet()->getDataPtr()->getTriangles();
		
		CArray<Coord> cPoints;
		cPoints.resize(mPoints.size());
		cPoints.assign(mPoints);

		CArray<Triangle> cTriangles;
		cTriangles.resize(mTriangles.size());
		cTriangles.assign(mTriangles);

		vertList.clear();
		faceList.clear();

		for (int i = 0; i < cPoints.size(); i++) {
			Coord point = cPoints[i];
			vertList.pushBack(point);
			min_box = min_box.minimum(point);
			max_box = max_box.maximum(point);
		}
	
		for (int i = 0; i < cTriangles.size(); i++) {
			faceList.pushBack(Vec3ui(cTriangles[i][0], cTriangles[i][1], cTriangles[i][2]));
		}
		
		uint padding = this->inPadding()->getData();
		Real dx = this->inSpacing()->getData();
		Vec3f unit(1, 1, 1);
		min_box -= padding * dx * unit;
		max_box += padding * dx * unit;

		ni = std::floor((max_box[0] - min_box[0]) / dx);
		nj = std::floor((max_box[1] - min_box[1]) / dx);
		nk = std::floor((max_box[2] - min_box[2]) / dx);
		
		origin = min_box;
		maxPoint = max_box;

		makeLevelSet();
	
		printf("Uniform grids: %f %f %f, %f %f %f, %f, %d %d %d, %d \n", origin[0], origin[1], origin[2], maxPoint[0], maxPoint[1], maxPoint[2], dx, ni, nj, nk, padding);
	}

	template<typename TDataType>
	void VolumeGenerator<TDataType>::load(std::string filename)
	{
		std::shared_ptr<TriangleSet<TDataType>> triSet = std::make_shared<TriangleSet<TDataType>>();
		triSet->loadObjFile(filename);

		this->inTriangleSet()->setDataPtr(triSet);

		this->inTriangleSet()->getDataPtr()->update();
	}

	// find distance x0 is from segment x1-x2
	static float point_segment_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2)
	{
		Vec3f dx(x2 - x1);
		double m2 = dx.normSquared();
		// find parameter value of closest point on segment
		float s12 = (float)((x2 - x0).dot(dx) / m2);
		if (s12 < 0) {
			s12 = 0;
		}
		else if (s12 > 1) {
			s12 = 1;
		}
		// and find the distance
		return (x0 - (s12*x1 + (1 - s12)*x2)).norm();
	}

	// find distance x0 is from triangle x1-x2-x3
	static float point_triangle_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, const Vec3f &x3)
	{
		// first find barycentric coordinates of closest point on infinite plane
		Vec3f x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
		float m13 = x13.normSquared(), m23 = x23.normSquared(), d = x13.dot(x23);
		float invdet = 1.f / std::max(m13*m23 - d * d, 1e-30f);
		float a = x13.dot(x03), b = x23.dot(x03);
		// the barycentric coordinates themselves
		float w23 = invdet * (m23*a - d * b);
		float w31 = invdet * (m13*b - d * a);
		float w12 = 1 - w23 - w31;
		if (w23 >= 0 && w31 >= 0 && w12 >= 0) { // if we're inside the triangle
			return (x0 - (w23*x1 + w31 * x2 + w12 * x3)).norm();
		}
		else { // we have to clamp to one of the edges
			if (w23 > 0) // this rules out edge 2-3 for us
				return std::min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x1, x3));
			else if (w31 > 0) // this rules out edge 1-3
				return std::min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x2, x3));
			else // w12 must be >0, ruling out edge 1-2
				return std::min(point_segment_distance(x0, x1, x3), point_segment_distance(x0, x2, x3));
		}
	}

	static void check_neighbour(const CArray<Vec3ui> &tri, const CArray<Vec3f> &vert,
		CArray3f &phi, CArray3i &closest_tri,
		const Vec3f &gx, int i0, int j0, int k0, int i1, int j1, int k1)
	{
		if (closest_tri(i1, j1, k1) >= 0) {
			unsigned int p, q, r; 
			Vec3ui trijk = tri[closest_tri(i1, j1, k1)];
			p = trijk[0];
			q = trijk[1];
			r = trijk[2];
			float d = point_triangle_distance(gx, vert[p], vert[q], vert[r]);
			if (d < phi(i0, j0, k0)) {
				phi(i0, j0, k0) = d;
				closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
			}
		}
	}

	static void sweep(const CArray<Vec3ui> &tri, const CArray<Vec3f> &x,
		CArray3f &phi, CArray3i &closest_tri, const Vec3f &origin, float dx,
		int di, int dj, int dk)
	{
		int i0, i1;
		if (di > 0) { 
			i0 = 1; i1 = phi.nx(); 
		}
		else { 
			i0 = phi.nx() - 2; i1 = -1; 
		}
		int j0, j1;
		if (dj > 0) { 
			j0 = 1; j1 = phi.ny(); 
		}
		else { 
			j0 = phi.ny() - 2; j1 = -1; 
		}
		int k0, k1;
		if (dk > 0) { 
			k0 = 1; k1 = phi.nz(); 
		}
		else { 
			k0 = phi.nz() - 2; k1 = -1; 
		}
		for (int k = k0; k != k1; k += dk) for (int j = j0; j != j1; j += dj) for (int i = i0; i != i1; i += di) {
			Vec3f gx(i*dx + origin[0], j*dx + origin[1], k*dx + origin[2]);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i, j, k - dk);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j, k - dk);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i, j - dj, k - dk);
			check_neighbour(tri, x, phi, closest_tri, gx, i, j, k, i - di, j - dj, k - dk);
		}
	}

	// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
	// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
	static int orientation(double x1, double y1, double x2, double y2, double &twice_signed_area)
	{
		twice_signed_area = y1 * x2 - x1 * y2;
		if (twice_signed_area > 0) return 1;
		else if (twice_signed_area < 0) return -1;
		else if (y2 > y1) return 1;
		else if (y2 < y1) return -1;
		else if (x1 > x2) return 1;
		else if (x1 < x2) return -1;
		else return 0; // only true when x1==x2 and y1==y2
	}

	// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
	// if true is returned, the barycentric coordinates are set in a,b,c.
	static bool point_in_triangle_2d(double x0, double y0,
		double x1, double y1, double x2, double y2, double x3, double y3,
		double& a, double& b, double& c)
	{
		x1 -= x0; x2 -= x0; x3 -= x0;
		y1 -= y0; y2 -= y0; y3 -= y0;
		int signa = orientation(x2, y2, x3, y3, a);
		if (signa == 0) return false;
		int signb = orientation(x3, y3, x1, y1, b);
		if (signb != signa) return false;
		int signc = orientation(x1, y1, x2, y2, c);
		if (signc != signa) return false;
		double sum = a + b + c;
		assert(sum != 0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
		a /= sum;
		b /= sum;
		c /= sum;
		return true;
	}

#define TRI_MIN(x, y, z) std::min(x, std::min(y, z))
#define TRI_MAX(x, y, z) std::max(x, std::max(y, z))

	template<typename TDataType>
	void VolumeGenerator<TDataType>::makeLevelSet()
	{
		const int exact_band = 1;
		Real dx = this->inSpacing()->getData();

		phi.resize(ni, nj, nk);
		phi.assign((ni + nj + nk)*dx);
		//TODO: phi.assign((ni + nj + nk)*dx); // upper bound on distance
		CArray3i closest_tri(ni, nj, nk);//CArray3i closest_tri(ni, nj, nk, -1);
		closest_tri.assign(-1);
		CArray3i intersection_count(ni, nj, nk); //CArray3i intersection_count(ni, nj, nk, 0); // intersection_count(i,j,k) is # of tri intersections in (i-1,i]x{j}x{k}
		intersection_count.assign(0);
		// we begin by initializing distances near the mesh, and figuring out intersection counts
		for (uint t = 0; t < faceList.size(); ++t) {
			uint p, q, r;
			p = faceList[t][0];
			q = faceList[t][1];
			r = faceList[t][2];
			// coordinates in grid to high precision
			double fip = ((double)vertList[p][0] - origin[0]) / dx;
			double fjp = ((double)vertList[p][1] - origin[1]) / dx;
			double fkp = ((double)vertList[p][2] - origin[2]) / dx;
			double fiq = ((double)vertList[q][0] - origin[0]) / dx; 
			double fjq = ((double)vertList[q][1] - origin[1]) / dx; 
			double fkq = ((double)vertList[q][2] - origin[2]) / dx;
			double fir = ((double)vertList[r][0] - origin[0]) / dx; 
			double fjr = ((double)vertList[r][1] - origin[1]) / dx; 
			double fkr = ((double)vertList[r][2] - origin[2]) / dx;
			// do distances nearby
			int i0 = clamp(int(TRI_MIN(fip, fiq, fir)) - exact_band, 0, ni - 1);
			int i1 = clamp(int(TRI_MAX(fip, fiq, fir)) + exact_band + 1, 0, ni - 1);
			int j0 = clamp(int(TRI_MIN(fjp, fjq, fjr)) - exact_band, 0, nj - 1);
			int j1 = clamp(int(TRI_MAX(fjp, fjq, fjr)) + exact_band + 1, 0, nj - 1);
			int k0 = clamp(int(TRI_MIN(fkp, fkq, fkr)) - exact_band, 0, nk - 1);
			int k1 = clamp(int(TRI_MAX(fkp, fkq, fkr)) + exact_band + 1, 0, nk - 1);
			for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) for (int i = i0; i <= i1; ++i) {
				Vec3f gx(i*dx + origin[0], j*dx + origin[1], k*dx + origin[2]);
				float d = point_triangle_distance(gx, vertList[p], vertList[q], vertList[r]);
				if (d < phi(i, j, k)) {
					phi(i, j, k) = d;
					closest_tri(i, j, k) = t;
				}
			}
			// and do intersection counts
			j0 = clamp((int)std::ceil(TRI_MIN(fjp, fjq, fjr)), 0, nj - 1);
			j1 = clamp((int)std::floor(TRI_MAX(fjp, fjq, fjr)), 0, nj - 1);
			k0 = clamp((int)std::ceil(TRI_MIN(fkp, fkq, fkr)), 0, nk - 1);
			k1 = clamp((int)std::floor(TRI_MAX(fkp, fkq, fkr)), 0, nk - 1);
			for (int k = k0; k <= k1; ++k) for (int j = j0; j <= j1; ++j) {
				double a, b, c;
				if (point_in_triangle_2d(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a, b, c)) {
					double fi = a * fip + b * fiq + c * fir; // intersection i coordinate
					int i_interval = int(std::ceil(fi)); // intersection is in (i_interval-1,i_interval]
					if (i_interval < 0) ++intersection_count(0, j, k); // we enlarge the first interval to include everything to the -x direction
					else if (i_interval < ni) ++intersection_count(i_interval, j, k);
					// we ignore intersections that are beyond the +x side of the grid
				}
			}
		}
		// and now we fill in the rest of the distances with fast sweeping
		for (unsigned int pass = 0; pass < 2; ++pass) {
			sweep(faceList, vertList, phi, closest_tri, origin, dx, +1, +1, +1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, -1, -1, -1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, +1, +1, -1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, -1, -1, +1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, +1, -1, +1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, -1, +1, -1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, +1, -1, -1);
			sweep(faceList, vertList, phi, closest_tri, origin, dx, -1, +1, +1);
		}
		// then figure out signs (inside/outside) from intersection counts
		for (int k = 0; k < nk; ++k) for (int j = 0; j < nj; ++j) {
			int total_count = 0;
			for (int i = 0; i < ni; ++i) {
				total_count += intersection_count(i, j, k);
				if (total_count % 2 == 1) { // if parity of intersections so far is odd,
					phi(i, j, k) = -phi(i, j, k); // we are inside the mesh
				}
			}
		}
	}

	DEFINE_CLASS(VolumeGenerator);
}