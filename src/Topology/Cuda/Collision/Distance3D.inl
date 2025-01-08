namespace dyno
{
	template<typename Real, typename Coord, DeviceType deviceType, typename IndexType>
	DYN_FUNC bool calculateSignedDistance2TriangleSet(ProjectedPoint3D<Real>& p3d, Coord point, Array<Coord, deviceType>& vertices, Array<TopologyModule::Triangle, deviceType>& indices, List<IndexType>& list, Real dHat)
	{
// 		auto PROJECT_INSIDE = [](const TPoint3D<Real> p, const TTriangle3D<Real> triangle) -> bool
// 			{
// 				TPlane3D<Real> plane(triangle.v[0], triangle.normal());
// 
// 				TPoint3D<Real> proj = p.project(plane);
// 
// 				typename TTriangle3D<Real>::Param tParam;
// 				bool bValid = triangle.computeBarycentrics(proj.origin, tParam);
// 				if (bValid)
// 				{
// 					return tParam.u > Real(0) && tParam.u < Real(1) && tParam.v > Real(0) && tParam.v < Real(1) && tParam.w > Real(0) && tParam.w < Real(1);
// 				}
// 				else
// 				{
// 					return false;
// 				}
// 			};

		Real eps = EPSILON;

		p3d.signed_distance = REAL_MAX;

		bool validate = false;

		TPoint3D<Real> pi = TPoint3D<Real>(point);
		for (uint j = 0; j < list.size(); j++)
		{
			TopologyModule::Triangle index = indices[list[j]];
			TTriangle3D<Real> tj(vertices[index[0]], vertices[index[1]], vertices[index[2]]);

			Coord nj = tj.normal();

			TPoint3D<Real> proj = pi.project(tj);
			Real d = (pi.origin - proj.origin).norm();

			//If the new triangle is closer, use the new triangle to update p3d
			if (glm::abs(d) < glm::abs(p3d.signed_distance) - eps)
			{
				p3d.signed_distance = d;
				p3d.point = proj.origin;
				p3d.id = list[j];

				validate = true;
			}
			//Otherwise, if the two triangles have relatively equal distances to the point, 
			//		further compare the distance from the point's projection to the corresponding triangle
			else if (glm::abs(d) < glm::abs(p3d.signed_distance) + eps)
			{
				typename TTriangle3D<Real>::Param tParam;

				TopologyModule::Triangle index_min = indices[p3d.id];
				TTriangle3D<Real> t_min(vertices[index_min[0]], vertices[index_min[1]], vertices[index_min[2]]);

				TPlane3D<Real> plane_j(tj.v[0], tj.normal());
				TPlane3D<Real> plane_min(t_min.v[0], t_min.normal());

				TPoint3D<Real> pn_j = pi.project(plane_j);
				TPoint3D<Real> pn_min = pi.project(plane_min);

				Real d_j = glm::abs(pn_j.distance(tj));
				Real d_min_j = glm::abs(pn_min.distance(t_min));

				if (d_j < d_min_j)
				{
					p3d.signed_distance = d;
					p3d.point = proj.origin;
					p3d.id = list[j];

					validate = true;
				}
			}
		}

		//Calculate the normal
		if (validate)
		{
			TopologyModule::Triangle index = indices[p3d.id];
			TTriangle3D<Real> tri(vertices[index[0]], vertices[index[1]], vertices[index[2]]);

			Coord n = pi.origin - p3d.point;
			n = n.norm() > EPSILON ? n.normalize() : tri.normal();

			//Check whether the point is located inside or not
			p3d.signed_distance = n.dot(tri.normal()) > 0 ? p3d.signed_distance : -p3d.signed_distance;
			p3d.signed_distance -= dHat;
			p3d.normal = n;
		}

		return validate;
	}



	template<typename Real, typename Coord, DeviceType deviceType, typename IndexType>
	DYN_FUNC bool calculateDistance2TriangleSet(ProjectedPoint3D<Real>& p3d, Coord point, Array<Coord, deviceType>& vertices, Array<TopologyModule::Triangle, deviceType>& indices, List<IndexType>& list, Real dHat)
	{
		p3d.signed_distance = REAL_MAX;

		bool validate = false;

		TPoint3D<Real> pi = TPoint3D<Real>(point);
		for (uint j = 0; j < list.size(); j++)
		{
			TopologyModule::Triangle index = indices[list[j]];
			TTriangle3D<Real> tj(vertices[index[0]], vertices[index[1]], vertices[index[2]]);

			Coord nj = tj.normal();

			TPoint3D<Real> proj = pi.project(tj);
			Real d = (pi.origin - proj.origin).norm();

			//If the new triangle is closer, use the new triangle to update p3d
			if (glm::abs(d) < glm::abs(p3d.signed_distance))
			{
				p3d.signed_distance = d;
				p3d.point = proj.origin;
				p3d.id = list[j];

				validate = true;
			}
		}

		//Calculate the normal
		if (validate)
		{
			TopologyModule::Triangle index = indices[p3d.id];
			TTriangle3D<Real> tri(vertices[index[0]], vertices[index[1]], vertices[index[2]]);

			Coord n = pi.origin - p3d.point;
			n = n.norm() > EPSILON ? n.normalize() : tri.normal();

			p3d.signed_distance -= dHat;
			p3d.normal = n;
		}

		return validate;
	}

	template <typename Coord,DeviceType deviceType, typename IndexType>
	DYN_FUNC void SO_ComputeObjectAndNormal(
		Coord& pobject,
		Coord& pnormal,
		Array<Coord, deviceType>& surf_points,
		Array<TopologyModule::Edge, deviceType>& edge,
		Array<TopologyModule::Triangle, deviceType>& surf_triangles,
		Array<TopologyModule::Tri2Edg, deviceType>& t2e,
		Array<Coord, deviceType>& edgeN,
		Array<Coord, deviceType>& vertexN,
		Coord ppos,
		IndexType surf_id)
	{
		int p = surf_triangles[surf_id][0];
		int q = surf_triangles[surf_id][1];
		int r = surf_triangles[surf_id][2];
		Coord p0 = surf_points[p];
		Coord p1 = surf_points[q];
		Coord p2 = surf_points[r];

		int eid00 = t2e[surf_id][0];
		int eid11 = t2e[surf_id][1];
		int eid22 = t2e[surf_id][2];
		int te00 = glm::min(p, q), te01 = glm::max(p, q);
		int te10 = glm::min(q, r), te11 = glm::max(q, r);
		int te20 = glm::min(r, p), te21 = glm::max(r, p);
		int eid0, eid1, eid2;
		if (edge[eid00][0] == te00&& edge[eid00][1] == te01)
		{
			eid0 = eid00;
			if (edge[eid11][0] == te10&& edge[eid11][1] == te11)
			{
				eid1 = eid11;
				eid2 = eid22;
			}
			else
			{
				eid1 = eid22;
				eid2 = eid11;
			}
		}
		else if (edge[eid00][0] == te10&& edge[eid00][1] == te11)
		{
			eid1 = eid00;
			if (edge[eid11][0] == te00&& edge[eid11][1] == te01)
			{
				eid0 = eid11;
				eid2 = eid22;
			}
			else
			{
				eid2 = eid11;
				eid0 = eid22;
			}
		}
		else if (edge[eid00][0] == te20 && edge[eid00][1] == te21)
		{
			eid2 = eid00;
			if (edge[eid11][0] == te00 && edge[eid11][1] == te01)
			{
				eid0 = eid11;
				eid1 = eid22;
			}
			else
			{
				eid1 = eid11;
				eid0 = eid22;
			}
		}

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
			//printf("111: %d %d %d, %d, %f %f %f \n", p, q, r, p, pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
		else if (s == 0 && t == 1)
		{
			pnormal = vertexN[r];
			//printf("222: %d %d %d, %d, %f %f %f \n", p, q, r, r, pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
		else if (s == 1 && t == 0)
		{
			pnormal = vertexN[q];
			//printf("333: %d %d %d, %d, %f %f %f \n", p, q, r, r, pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
		else if (s == 0 && t < 1)
		{
			pnormal = edgeN[eid2];
			//printf("444: %d %d %d, %d %d %d, %f %f %f \n", p, q, r, eid2, edge[eid2][0], edge[eid2][1], pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
		else if (s < 1 && t == 0)
		{
			pnormal = edgeN[eid0];
			//printf("555: %d %d %d, %d %d %d, %f %f %f \n", p, q, r, eid0, edge[eid0][0], edge[eid0][1], pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
		else if (s + t == 1)
		{
			pnormal = edgeN[eid1];
			//printf("666: %d %d %d, %d %d %d, %f %f %f \n", p, q, r, eid1, edge[eid1][0], edge[eid1][1], pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
		else
		{
			pnormal = (p1 - p0).cross(p2 - p0);
			pnormal.normalize();
			//printf("777: %d %d %d, %f %f %f \n", p, q, r, pnormal[0], pnormal[1], pnormal[2]);
			return;
		}
	}
	template<typename Real, typename Coord, DeviceType deviceType, typename IndexType>
	DYN_FUNC bool calculateSignedDistance2TriangleSetFromNormal(
		ProjectedPoint3D<Real>& p3d,
		Coord point,
		Array<Coord, deviceType>& vertices,
		Array<TopologyModule::Edge, deviceType>& edges,
		Array<TopologyModule::Triangle, deviceType>& triangles,
		Array<TopologyModule::Tri2Edg, deviceType>& t2e,
		Array<Coord, deviceType>& edgeNormal,
		Array<Coord, deviceType>& vertexNormal,
		List<IndexType>& list,
		Real dHat)
	{
		Real eps = 10 * EPSILON;

		p3d.signed_distance = REAL_MAX;

		bool validate = false;
		for (uint j = 0; j < list.size(); j++)
		{
			Coord pnormal(0), pobject(0);

			SO_ComputeObjectAndNormal(pobject, pnormal, vertices, edges, triangles, t2e, edgeNormal, vertexNormal, point, list[j]);

			Real sign = (point - pobject).dot(pnormal) > Real(0) ? Real(1) : Real(-1);
			Real d = sign * (point - pobject).norm();

			//If the new triangle is closer, use the new triangle to update p3d
			if (glm::abs(d) < glm::abs(p3d.signed_distance) - eps)
			{
				p3d.signed_distance = d;
				p3d.point = pobject;
				p3d.normal = pnormal;
				p3d.id = list[j];

				validate = true;
			}
		}
		return validate;
	}

}
