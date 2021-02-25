#pragma once
#include "Primitive3D.h"


namespace dyno
{
	typedef typename TSphere3D<Real> Sphere3D;
	typedef typename TOrientedBox3D<Real> Box3D;

	template <typename Real> class TNeighborConstraints;

	template<typename Real>
	class TNeighborConstraints 
	{
	public:
		

		DYN_FUNC TNeighborConstraints(int a, int b, int type, Coord3D p1, Coord3D p2, Coord3D n1, Coord3D n2)
		{
			idx1 = a;
			idx2 = b;
			constraint_type = type;
			pos1 = p1;
			pos2 = p2;
			normal1 = n1;
			normal2 = n2;
		}
		DYN_FUNC ~TNeighborConstraints() {}
		

		Real s1, s2, s3, s4; //reserved scalar
		Coord3D v1, v2, v3, v4; //reserved vector

		int idx1, idx2;
		int constraint_type;
		Coord3D pos1, pos2;
		Coord3D normal1, normal2;
		

	};


#ifdef PRECISION_FLOAT
	template class TNeighborConstraints <float>;
	
#else
	template class TNeighborConstraints <double>;
#endif
}

