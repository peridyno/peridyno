#pragma once
#include "Primitive3D.h"
#include "Module/ComputeModule.h"

namespace dyno
{
	typedef typename TSphere3D<Real> Sphere3D;
	typedef typename TOrientedBox3D<Real> Box3D;

	template <typename Real> class TNeighborConstraints;

	template<typename Real>
	class TNeighborConstraints 
	{
		
	public:
		
		DYN_FUNC TNeighborConstraints()
		{
			idx1 = idx2 = 0;
			//s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = s9 = 0.0f;
//			tag0 = tag1 = tag2 = tag3 = tag4 = tag5 = tag6 = tag7 = 0;
			constraint_type = -10;
			//s7 = -1.0f;
			//mass1 = 0.0f;
			local_index_1 = -1;
			local_index_2 = -1;
//			i3 = i4 = i5 = i6 = -1;
		};
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
		
		Real inter_distance = 0.0f;//inter_dist
		Real s1, s2, s3, s5, s6, s7, s8, s9; //reserved scalar
		Real viscosity_normal = 0.0f;
		Real viscosity_tangential = 0.0f;
		bool two_scale_constraint_tag = false;
		Coord3D v1, v2, v3, v4; //reserved vector
		int local_index_1 = -1;
		int local_index_2 = -1;
		int two_scale_type = 0; //1:local  3:global  else: not two scale  
		int idx1_local = -1;
		int idx2_local = -1;
		int i6 = -1;
		int i7 = -1;

		int idx1, idx2;
		int constraint_type;

		
		Coord3D pos1, pos2;
		Coord3D normal1, normal2;
		
		Real mass_1, mass_2;

		static const int type_boundary_constraint = 0;
		static const int type_nointerpenetration_bodies = 1;
		static const int type_friction = 5;
		static const int type_fluid_stickiness = 6;
		static const int type_fluid_slippiness = 7;
		static const int type_nointerpenetration_fluid = 8;
		static const int type_nointerpenetration_local = 10;
		static const int type_disable = 11;


		static const int two_scale_local = 1;
		static const int two_scale_global = 3;
		static const int not_two_scale = 0;
	};


#ifdef PRECISION_FLOAT
	template class TNeighborConstraints <float>;
	
#else
	template class TNeighborConstraints <double>;
#endif
}

