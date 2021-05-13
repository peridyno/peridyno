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

#ifndef __DISTANCEFIELD3D_H__
#define __DISTANCEFIELD3D_H__

#include <vector>
#include <string>
#include "Vec.h"     //’‚¿Ô «vec.h∞…£ø£ø£ø
#include "makelevelset3.h"

using namespace std;

namespace mfd {

class DistanceField3D {
   public:

      DistanceField3D(string filename,float dx_grid,int padding_grid);

      DistanceField3D(string filename);

      DistanceField3D(const Vec3d p0_, const Vec3d p1_, int nbx_, int nby_, int nbz_, bool inv_ = false);

      virtual ~DistanceField3D();
      

      void WriteToFile(string filename); 
	  void ReadSDF(string filename);

	  double DistanceToBox(Vec3d& pos, Vec3d& lo, Vec3d& hi);
	  double DistanceToSqure(Vec3d& pos, Vec3d& lo, Vec3d& hi, int axis);
	  double DistanceToSegment(Vec3d& pos, Vec3d& lo, Vec3d& hi);
	  double DistanceToCylinder(Vec3d& pos, Vec3d& center, double radius, double height, int axis);

	  void DistanceFieldToBox(Vec3d& lo, Vec3d& hi, bool inverted);
	  void DistanceFieldToCylinder(Vec3d& center, double radius, double height, int axis, bool inverted);
	 
	  void RotationDistance(double alpha);

	  void SetDistance(int index, double d) {
		  distances[index] = d;
	  }

      void SetDistance(int i, int j, int k, double d) {
         const int index = i+nbx*(j+nby*k);
         distances[index] = d;
      }

	  double GetDistance(int i, int j, int k) {
         return distances[i+nbx*(j+nby*k)];
      }

      void GetDistance(const Vec3d &p, double &d);

      void GetDistance(const Vec3d &p, double &d, Vec3d &g);

      inline double Lerp(double a, double b, double alpha) const {
         return (1.0-alpha)*a + alpha *b;
      }

      inline void Initialize();

      inline void Normalize();

      void Translate(const Vec3d &t);
      void Scale(const Vec3d &s);
      void Invert();


	  Vec3d p0; // lower left front corner
	  Vec3d p1; // upper right back corner
	  Vec3d h;  // single cell sizes
	  Vec3d invh;   // inverse of single cell sizes
	  int nbx,nby,nbz; // number of cells in all dimensions

	  double * distances;
	  double * weights;
      Vec3d * positions;

      int _list;//this variable is for what?
	  bool bInvert;//this variable is for what?
};

}

#endif
