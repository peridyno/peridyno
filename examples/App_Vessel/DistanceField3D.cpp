/******************************************************************************
Copyright (c) 2007 Bart Adams (bart.adams@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. The authors shall be
acknowledged in scientific publications resulting from using the Software
by referencing the ACM SIGGRAPH 2007 paper "Adaptively Sampled Particle
Fluids".

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/

//#include <windows.h>
//#include <GL/gl.h>
#include <cmath>
//#include <cdouble>
#include <fstream>
#include <sstream>
#include <iostream>
#include <queue>
#include <vector>
#include <limits>

#include "Array/Array3D.h"
#include "DistanceField3D.h"

using namespace mfd;

double mysign(double x) 
{
   if (x<0) return -1;
   else return +1;
}

//DistanceField3D::DistanceField3D(string filename, float dx_grid, int padding_grid)
//{
//	if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj")) 
//	{
//		std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
//		exit(-1);
//	}
//	if (padding_grid < 1) padding_grid = 1;
//
//	//start with a massive inside out bound box.
//	Vec3f min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
//		max_box(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
//
//
//	//start reading .obj file
//	std::cout << "Reading data.\n";
//	std::ifstream infile(filename);
//	if (!infile) {
//		std::cerr << "Failed to open. Terminating.\n";
//		exit(-1);
//	}
//
//	std::string line;
//	std::vector<Vec3f> vertList;
//	std::vector<Vec3ui> faceList;
//	while (!infile.eof()) 
//	{
//		std::getline(infile, line);
//
//		if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn")) 
//		{
//			std::stringstream data(line);
//			char c;
//			Vec3f point;
//			data >> c >> point[0] >> point[1] >> point[2];
//			vertList.push_back(point);
//			update_minmax(point, min_box, max_box);
//		}
//		else if (line.substr(0, 1) == std::string("f")) 
//		{
//			std::stringstream data(line);
//			char c, c11, c12, c21, c22, c31, c32;
//			int v0, v1, v2, mtl0, mtl1, mtl2;
//			data >> c >> v0 >> c11 >> c12 >> mtl0 >> v1 >> c21 >> c22 >> mtl1 >> v2 >> c31 >> c32 >> mtl2;
//			faceList.push_back(Vec3ui(v0 - 1, v1 - 1, v2 - 1));
//		}
//	}
//	infile.close();
//
//	std::cout << "Read in " << vertList.size() << " vertices and " << faceList.size() << " faces." << std::endl;
//
//
//	//Add padding around the box.
//	Vec3f unit(1, 1, 1);
//	min_box -= padding_grid * dx_grid * unit;
//	max_box += padding_grid * dx_grid * unit;
//	Vec3ui sizes = Vec3ui((max_box - min_box) / dx_grid);
//
//	std::cout << "Bound box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;
//
//
//	//start computing signed distance field
//	std::cout << "Computing signed distance field.\n";
//	Array3f phi_grid;
//	make_level_set3(faceList, vertList, min_box, dx_grid, sizes[0], sizes[1], sizes[2], phi_grid);
//	std::cout << "Computing signed distance field.\n";
//
//
//	//store sdf
//	ifstream input(filename.c_str(), ios::in);
//	int xx = phi_grid.ni;
//	int yy = phi_grid.nj;
//	int zz = phi_grid.nk;
//	p0[0] = min_box[0];
//	p0[1] = min_box[1];
//	p0[2] = min_box[2];
//	double t_h=dx_grid;
//	p1[0] = p0[0] + t_h * (xx-1);
//	p1[1] = p0[1] + t_h * (yy-1);
//	p1[2] = p0[2] + t_h * (zz-1);
//
//	cout << "Bunny: the number of vertex: " << xx << ", " << yy << ", " << zz << endl;
//	cout << "Bunny: the left lower vertex: " << p0[0] << ", " << p0[1] << ", " << p0[2] << endl;
//	cout << "Bunny: the upper right vertex: " << p1[0] << ", " << p1[1] << ", " << p1[2] << endl;
//
//	nbx = xx;
//	nby = yy;
//	nbz = zz;
//	h[0] = t_h;
//	h[1] = t_h;
//	h[2] = t_h;
//
//	distances = new double[(nbx) * (nby) * (nbz)];
//	invh = Vec3d(1.0f / h[0], 1.0f / h[1], 1.0f / h[2]);
//	for (int k = 0; k < zz; k++) {
//		for (int j = 0; j < yy; j++) {
//			for (int i = 0; i < xx; i++) {
//				SetDistance((i), (j), (k), phi_grid(i, j, k));
//			}
//		}
//	}
//	weights = NULL;
//	positions = NULL;
//
//	_list = -1;
//	bInvert = false;
//}

DistanceField3D::DistanceField3D(string filename) 
{
	ReadSDF(filename);
	//_list = -1;
	//bInvert = false;
}

DistanceField3D::DistanceField3D(const Vec3d p0_, const Vec3d p1_, int nbx_, int nby_, int nbz_, bool inv_) 
: p0(p0_), p1(p1_), nbx(nbx_), nby(nby_), nbz(nbz_) 
{
	h = (p1-p0)*Vec3d(1.0f/ double(nbx-1), 1.0f/ double(nby-1), 1.0f/ double(nbz-1));
	cout << h << std::endl;
	invh = Vec3d(1.0f/h[0], 1.0f/h[1], 1.0f/h[2]);
	distances = new double[(nbx)*(nby)*(nbz)];
	weights = new double[(nbx)*(nby)*(nbz)];
	positions = new Vec3d[(nbx)*(nby)*(nbz)];
	//_list = -1;
	//bInvert = inv_;
}

DistanceField3D::~DistanceField3D() 
{
    delete[] distances;
    distances = NULL;
    delete[] weights;
    weights = NULL;
    delete[] positions;
    positions = NULL;
}


void DistanceField3D::Initialize() 
{
	for (int i=0; i<nbx; i++) {
		for (int j=0; j<nby; j++) {
			for (int k=0; k<nbz; k++) {
				const int index = i+nbx*(j+nby*k);
				weights[index] = 0;
				distances[index] = 0;
			}
		}
	}
}

void DistanceField3D::Normalize() 
{
	for (int i=0; i<nbx; i++) {
		for (int j=0; j<nby; j++) {
			for (int k=0; k<nbz; k++) {
				const Vec3d p = p0+Vec3d(i,j,k)*h;
				const int index = i+nbx*(j+nby*k);
				const double weight = weights[index];
				if (weight<0.00000000001f) distances[index] = -h[0]/8.0f;
				else {
					double inv_weight = 1.0f/weight;
					distances[index] = distances[index] * inv_weight;
// 					// this is the final field
// 					distances[index] = (dist- Distance(pos, p));
				}
			}
		}
	}
}

void DistanceField3D::GetDistance(const Vec3d &p, double &d)
{
	// get cell and lerp values
	Vec3d fp = (p-p0)*invh;
	const int i = (int)floor(fp[0]);
	const int j = (int)floor(fp[1]);
	const int k = (int)floor(fp[2]);
	//cout << "get distance: " << i << " " << j << " " << k << std::endl;
	if (i<0 || i>=nbx-1 || j<0 || j>=nby-1 || k<0 || k>=nbz-1) 
	{
		d =FLT_MAX;
		return;
	}
	Vec3d ip(i,j,k);
	//ip.bound(Vec3d(0,0,0), Vec3d(nbx-1,nby-1,nbz-1));

	Vec3d alphav = fp-ip;
	double alpha = alphav[0];
	double beta = alphav[1];
	double gamma = alphav[2];

	const double d000 = GetDistance(i,j,k);
	const double d100 = GetDistance(i+1,j,k);
	const double d010 = GetDistance(i,j+1,k);
	const double d110 = GetDistance(i+1,j+1,k);
	const double d001 = GetDistance(i,j,k+1);
	const double d101 = GetDistance(i+1,j,k+1);
	const double d011 = GetDistance(i,j+1,k+1);
	const double d111 = GetDistance(i+1,j+1,k+1);

	const double dx00 = (1.0f-alpha) * d000 + alpha * d100;
	const double dx10 = (1.0f-alpha) * d010 + alpha * d110;

	const double dxy0 = (1.0f-beta) * dx00 + beta * dx10;

	const double dx01 = (1.0f-alpha) * d001 + alpha * d101;
	const double dx11 = (1.0f-alpha) * d011 + alpha * d111;

	const double dxy1 = (1.0f-beta) * dx01 + beta * dx11;

	d = (1.0f-gamma) * dxy0 + gamma * dxy1;
}

void DistanceField3D::GetDistance(const Vec3d &p, double &d, Vec3d &g)
{
	// get cell and lerp values
	Vec3d fp = (p-p0)*invh;
	const int i = (int)floor(fp[0]);
	const int j = (int)floor(fp[1]);
	const int k = (int)floor(fp[2]);
	if (i<0 || i >= nbx - 1 || j<0 || j >= nby - 1 || k<0 || k >= nbz - 1) 
	{
		d = FLT_MAX;
		g = Vec3d(0.0, 0.0, 0.0);
		return;
	}
	Vec3d ip(i,j,k);

	Vec3d alphav = fp-ip;
	double alpha = alphav[0];
	double beta = alphav[1];
	double gamma = alphav[2];

	const double d000 = GetDistance(i,j,k);
	const double d100 = GetDistance(i+1,j,k);
	const double d010 = GetDistance(i,j+1,k);
	const double d110 = GetDistance(i+1,j+1,k);
	const double d001 = GetDistance(i,j,k+1);
	const double d101 = GetDistance(i+1,j,k+1);
	const double d011 = GetDistance(i,j+1,k+1);
	const double d111 = GetDistance(i+1,j+1,k+1);

	const double dx00 = Lerp(d000, d100, alpha);
	const double dx10 = Lerp(d010, d110, alpha);
	const double dxy0 = Lerp(dx00, dx10, beta);

	const double dx01 = Lerp(d001, d101, alpha);
	const double dx11 = Lerp(d011, d111, alpha);
	const double dxy1 = Lerp(dx01, dx11, beta);

	const double d0y0 = Lerp(d000, d010, beta);
	const double d0y1 = Lerp(d001, d011, beta);
	const double d0yz = Lerp(d0y0, d0y1, gamma);

	const double d1y0 = Lerp(d100, d110, beta);
	const double d1y1 = Lerp(d101, d111, beta);
	const double d1yz = Lerp(d1y0, d1y1, gamma);

	const double dx0z = Lerp(dx00, dx01, gamma);
	const double dx1z = Lerp(dx10, dx11, gamma);

	g[0] = d0yz - d1yz;
	g[1] = dx0z - dx1z;
	g[2] = dxy0 - dxy1;

	double length = g.norm();
	if (length<0.0001) g = Vec3d(0.0, 0.0, 0.0);

	d = (1.0-gamma) * dxy0 + gamma * dxy1;
}

void DistanceField3D::WriteToFile(string filename) 
{
   ofstream output(filename.c_str(), ios::out|ios::binary);
   output.write((char*)&p0[0], sizeof(double));
   output.write((char*)&p0[1], sizeof(double));
   output.write((char*)&p0[2], sizeof(double));
   output.write((char*)&p1[0], sizeof(double));
   output.write((char*)&p1[1], sizeof(double));
   output.write((char*)&p1[2], sizeof(double));
   output.write((char*)&h[0], sizeof(double));
   output.write((char*)&h[1], sizeof(double));
   output.write((char*)&h[2], sizeof(double));
   output.write((char*)&nbx, sizeof(int));
   output.write((char*)&nby, sizeof(int));
   output.write((char*)&nbz, sizeof(int));
   for (int i=0; i<=nbx; i++) {
      for (int j=0; j<=nby; j++) {
         for (int k=0; k<=nbz; k++) {
			 double dist = GetDistance(i,j,k);
            output.write((char*)&dist, sizeof(double));
         }
      }
   }
   output.close();
}

void DistanceField3D::Translate(const Vec3d &t) 
{
   p0+=t;
   p1+=t;
}

void DistanceField3D::Scale(const Vec3d &s)
{
   p0[0] *= s[0];
   p0[1] *= s[1];
   p0[2] *= s[2];
   p1[0] *= s[0];
   p1[1] *= s[1];
   p1[2] *= s[2];
   h[0] *= s[0];
   h[1] *= s[1];
   h[2] *= s[2];
   invh = Vec3d(1.0f/h[0], 1.0f/h[1], 1.0f/h[2]);
   for (int i=0; i<(nbx)*(nby)*(nbz); i++) {
      distances[i]=s[0]*distances[i];
   }
}

void DistanceField3D::Invert() 
{
   const int nb = (nbx)*(nby)*(nbz);
   for (int i=0; i<nb; i++) distances[i] = -distances[i];
}

void DistanceField3D::ReadSDF(string filename)
{
	cout << "now it is: " << filename.c_str() << endl;
	ifstream input(filename.c_str(), ios::in);
	int xx, yy, zz;
	input >> xx;
	input >> yy;
	input >> zz;
	input >> p0[0];
	input >> p0[1];
	input >> p0[2];
	double t_h;
	input >> t_h;
	p1[0] = p0[0]+t_h*(xx-1);
	p1[1] = p0[1]+t_h*(yy-1);
	p1[2] = p0[2]+t_h*(zz-1);

	cout << "Bunny: the number of vertex: " << xx << ", " << yy << ", " << zz << endl;
	cout << "Bunny: the left lower vertex: " << p0[0] << ", " << p0[1] << ", " << p0[2] << endl;
	cout << "Bunny: the upper right vertex: " << p1[0] << ", " << p1[1] << ", " << p1[2] << endl;

	nbx = xx;
	nby = yy;
	nbz = zz;
	h[0] = t_h;
	h[1] = t_h;
	h[2] = t_h;

	//int idd = 0;
	distances = new double[(nbx)*(nby)*(nbz)];
	invh = Vec3d(1.0f/h[0], 1.0f/h[1], 1.0f/h[2]);
	for (int k=0; k<zz; k++) {
		for (int j=0; j<yy; j++) {
			for (int i=0; i<xx; i++) 
			{
				double dist;
				input >> dist;
				SetDistance((i), (j), (k),dist);
			}
		}
	}
	input.close();
	weights = NULL;
	positions = NULL;

	cout << "read data successful" << endl;
}

double mfd::DistanceField3D::DistanceToBox(Vec3d& pos, Vec3d& lo, Vec3d& hi)
{
	Vec3d corner0 = Vec3d(lo[0], lo[1], lo[2]);
	Vec3d corner1 = Vec3d(hi[0], lo[1], lo[2]);
	Vec3d corner2 = Vec3d(hi[0], hi[1], lo[2]);
	Vec3d corner3 = Vec3d(lo[0], hi[1], lo[2]);
	Vec3d corner4 = Vec3d(lo[0], lo[1], hi[2]);
	Vec3d corner5 = Vec3d(hi[0], lo[1], hi[2]);
	Vec3d corner6 = Vec3d(hi[0], hi[1], hi[2]);
	Vec3d corner7 = Vec3d(lo[0], hi[1], hi[2]);
	//double dist0 = dist(pos, corner0);
	//double dist1 = dist(pos, corner1);
	//double dist2 = dist(pos, corner2);
	//double dist3 = dist(pos, corner3);
	//double dist4 = dist(pos, corner4);
	//double dist5 = dist(pos, corner5);
	//double dist6 = dist(pos, corner6);
	//double dist7 = dist(pos, corner7);
	if (pos[0] < hi[0] && pos[0] > lo[0] && pos[1] < hi[1] && pos[1] > lo[1] && pos[2] < hi[2] && pos[2] > lo[2])
	{
		double distx = min(abs(pos[0] - hi[0]), abs(pos[0] - lo[0]));
		double disty = min(abs(pos[1] - hi[1]), abs(pos[1] - lo[1]));
		double distz = min(abs(pos[2] - hi[2]), abs(pos[2] - lo[2]));
		double mindist = min(distx, disty);
		mindist = min(mindist, distz);
		return mindist;
	}
	else
	{
		double distx1 = DistanceToSqure(pos, corner0, corner7, 0);
		double distx2 = DistanceToSqure(pos, corner1, corner6, 0);
		double disty1 = DistanceToSqure(pos, corner0, corner5, 1);
		double disty2 = DistanceToSqure(pos, corner3, corner6, 1);
		double distz1 = DistanceToSqure(pos, corner0, corner2, 2);
		double distz2 = DistanceToSqure(pos, corner4, corner6, 2);
		return -min(min(min(distx1, distx2), min(disty1, disty2)), min(distz1, distz2));
	}
}

double mfd::DistanceField3D::DistanceToSqure(Vec3d& pos, Vec3d& lo, Vec3d& hi, int axis)
{
	Vec3d n;
	Vec3d corner1, corner2, corner3, corner4;
	Vec3d loCorner, hiCorner, p;
	switch (axis)
	{
	case 0:
		corner1 = Vec3d(lo[0], lo[1], lo[2]);
		corner2 = Vec3d(lo[0], hi[1], lo[2]);
		corner3 = Vec3d(lo[0], hi[1], hi[2]);
		corner4 = Vec3d(lo[0], lo[1], hi[2]);
		n = Vec3d(1.0f, 0.0f, 0.0f);

		loCorner = Vec3d(lo[1], lo[2], 0.0f);
		hiCorner = Vec3d(hi[1], hi[2], 0.0f);
		p = Vec3d(pos[1], pos[2], 0.0f);
		break;
	case 1:
		corner1 = Vec3d(lo[0], lo[1], lo[2]);
		corner2 = Vec3d(lo[0], lo[1], hi[2]);
		corner3 = Vec3d(hi[0], lo[1], hi[2]);
		corner4 = Vec3d(hi[0], lo[1], lo[2]);
		n = Vec3d(0.0f, 1.0f, 0.0f);

		loCorner = Vec3d(lo[0], lo[2], 0.0f);
		hiCorner = Vec3d(hi[0], hi[2], 0.0f);
		p = Vec3d(pos[0], pos[2], 0.0f);
		break;
	case 2:
		corner1 = Vec3d(lo[0], lo[1], lo[2]);
		corner2 = Vec3d(hi[0], lo[1], lo[2]);
		corner3 = Vec3d(hi[0], hi[1], lo[2]);
		corner4 = Vec3d(lo[0], hi[1], lo[2]);
		n = Vec3d(0.0f, 0.0f, 1.0f);

		loCorner = Vec3d(lo[0], lo[1], 0.0f);
		hiCorner = Vec3d(hi[0], hi[1], 0.0f);
		p = Vec3d(pos[0], pos[1], 0.0f);
		break;
	}

	double dist1 = DistanceToSegment(pos, corner1, corner2);
	double dist2 = DistanceToSegment(pos, corner2, corner3);
	double dist3 = DistanceToSegment(pos, corner3, corner4);
	double dist4 = DistanceToSegment(pos, corner4, corner1);
	double dist5 = abs(n.dot(pos - corner1));
	if (p[0] < hiCorner[0] && p[0] > loCorner[0] && p[1] < hiCorner[1] && p[1] > loCorner[1])
		return dist5;
	else
		return min(min(dist1, dist2), min(dist3, dist4));
}

double mfd::DistanceField3D::DistanceToSegment(Vec3d& pos, Vec3d& lo, Vec3d& hi)
{
	Vec3d seg = hi - lo;
	Vec3d edge1 = pos - lo;
	Vec3d edge2 = pos - hi;
	if (edge1.dot(seg) < 0.0f)
	{
		return edge1.norm();
	}
	if (edge2.dot(-seg) < 0.0f)
	{
		return edge2.norm();
	}
	double length1 = edge1.normSquared();
	seg = seg.normalize();
	double length2 = edge1.dot(seg);
	return sqrt(length1 - length2 * length2);
}

double mfd::DistanceField3D::DistanceToCylinder(Vec3d& pos, Vec3d& center, double radius, double height, int axis)
{
	double distR;
	double distH;
	switch (axis)
	{
	case 0:
		distH = abs(pos[0] - center[0]);
		distR = Vec3d(0.0f, pos[1] - center[1], pos[2] - center[2]).norm();
		break;
	case 1:
		distH = abs(pos[1] - center[1]);
		distR = Vec3d(pos[0] - center[0], 0.0f, pos[2] - center[2]).norm();
		break;
	case 2:
		distH = abs(pos[2] - center[2]);
		distR = Vec3d(pos[0] - center[0], pos[1] - center[1], 0.0f).norm();
		break;
	}

	double halfH = height / 2.0f;
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
		double l1 = distR - radius;
		double l2 = distH - halfH;
		return sqrt(l1*l1 + l2 * l2);
	}
}

void mfd::DistanceField3D::DistanceFieldToBox(Vec3d& lo, Vec3d& hi, bool inverted)
{
	int sign = inverted ? 1.0f : -1.0f;
	for (int k = 0; k < nbz; k++) {
		for (int j = 0; j < nby; j++) {
			for (int i = 0; i < nbx; i++) {
				Vec3d p = p0 + Vec3d(i, j, k)*h;
				double dist = sign * DistanceToBox(p, lo, hi);
				SetDistance(i, j, k, dist);
			}
		}
	}
}

void mfd::DistanceField3D::DistanceFieldToCylinder(Vec3d& center, double radius, double height, int axis, bool inverted)
{
	int sign = inverted ? -1.0f : 1.0f;
	//bInvert = sign;
	for (int k = 0; k < nbz; k++) {
		for (int j = 0; j < nby; j++) {
			for (int i = 0; i < nbx; i++) {
				Vec3d p = p0 + Vec3d(i, j, k)*h;
				double dist = sign * DistanceToCylinder(p, center, radius, height, axis);
				SetDistance(i, j, k, dist);
			}
		}
	}
}

void mfd::DistanceField3D::RotationDistance(double alpha)
{
	vector<double> dist(nbx*nby*nbz);

	for (int k = 0; k < nbz; k++) {
		for (int j = 0; j < nby; j++) {
			for (int i = 0; i < nbx; i++) {
				int index = i + j * nbx + k * nbx*nby;
				Vec3d pnew = p0 + Vec3d(i, j, k)*h;
				Vec3d pold(pnew[0] * cos(alpha) - pnew[1] * sin(alpha), pnew[0] * sin(alpha) + pnew[1] * cos(alpha), pnew[2]);
				double d;
				GetDistance(pold, d);
				dist[index] = d;
				//cout << "the i j k pnew pold d is: " << i << " " << j << " " << k << " " << pnew << " " << pold << " " << d << std::endl;
			}
		}
	}
	for (int k = 0; k < nbz; k++) {
		for (int j = 0; j < nby; j++) {
			for (int i = 0; i < nbx; i++) {
				int index = i + j * nbx + k * nbx*nby;
				SetDistance(i, j, k, dist[index]);
			}
		}
	}
}