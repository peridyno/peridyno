/**
 * Copyright 2021 Yue Chang
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Primitive/Primitive3D.h"

namespace dyno
{
	/**
	 * @brief Calculate the intersection area between a sphere and a triangle by using the domain decompsotion algorithm.
	 * 			For more details, please refer to Section 4.1 of [Chang et al. 2020]: "Semi-analytical Solid Boundary Conditions for Free Surface Flows"
	 * 
	 * @param pt center of the sphere
	 * @param triangle trianlge
	 * @param r radius of the sphere
	 * @return the intersection area
	 */
	template<typename Real>
	DYN_FUNC inline Real calculateIntersectionArea(const TPoint3D<Real>& pt, const TTriangle3D<Real>& triangle, const Real& R)
	{
		typedef typename Vector<Real, 3> Coord3D;

		Real R2 = R * R;

		if (triangle.area() < EPSILON) {
			return Real(0);
		}

		TPlane3D<Real> plane = TPlane3D<Real>(triangle.v[0], triangle.normal());

		Real d2 = pt.distanceSquared(plane);

		if (d2 >= R2) return Real(0);

		//case 0
		if (abs(pt.distance(triangle)) > R)
			return Real(0);

		Real r = sqrt(R2 - d2);

		TPoint3D<Real> center = pt.project(plane);

		uint nv = 0;
		for (uint j = 0; j < 3; j++)
		{
			if ((triangle.v[j] - center.origin).norm() <= r)
			{
				nv++;
			}
		}

		//case 9
		if (nv == 3)
			return triangle.area();

		//case 8
		if (nv == 2)
		{
			Real ret = Real(0);
			uint outsideId = 0;
			for (uint j = 0; j < 3; j++)
			{
				if ((triangle.v[j] - center.origin).norm() > r)
				{
					outsideId = j;
					break;
				}
			}

			Coord3D v0 = triangle.v[outsideId];
			Coord3D v1 = triangle.v[(outsideId + 1) % 3];
			Coord3D v2 = triangle.v[(outsideId + 2) % 3];

			Coord3D dir01 = v1 - v0;	dir01.normalize();
			Coord3D dir02 = v2 - v0;	dir02.normalize();

			Line3D line1 = Line3D(v0, dir01);
			Line3D line2 = Line3D(v0, dir02);

			Point3D p1 = center.project(line1);
			Point3D p2 = center.project(line2);

			Real l1 = sqrt(r * r - center.distanceSquared(p1));
			Real l2 = sqrt(r * r - center.distanceSquared(p2));

			Coord3D s1 = p1.origin - l1 * dir01;
			Coord3D s2 = p2.origin - l2 * dir02;

			ret += TTriangle3D<Real>(v0, v1, v2).area();
			ret -= TTriangle3D<Real>(v0, s1, s2).area();

			Real d1 = center.distance(Segment3D(s1, s2));
			Real d10 = 0.5 * (s1 - s2).norm();
			Real angle = asin(d10 / r);
			Real secArea = angle * r * r - d1 * d10;

			ret += secArea;

			return maximum(ret, Real(0));
		}

		//case 6 and 7
		if (nv == 1)
		{
			Real ret = Real(0);
			uint insideId = 0;
			for (uint j = 0; j < 3; j++)
			{
				if ((triangle.v[j] - center.origin).norm() <= r)
				{
					insideId = j;
					break;
				}
			}

			Coord3D v0 = triangle.v[insideId];
			Coord3D v1 = triangle.v[(insideId + 1) % 3];
			Coord3D v2 = triangle.v[(insideId + 2) % 3];

			//Calculate the intersection points between the circle and v0-v1/v0-v2
			Coord3D dir01 = v1 - v0;	dir01.normalize();
			Coord3D dir02 = v2 - v0;	dir02.normalize();
			Coord3D dir12 = v2 - v1;	dir12.normalize();

			Line3D line01 = Line3D(v0, dir01);
			Line3D line02 = Line3D(v0, dir02);

			Point3D p1 = center.project(line01);
			Point3D p2 = center.project(line02);

			Real d1 = center.distance(p1);
			Real d2 = center.distance(p2);

			Real l1 = sqrt(r * r - d1 * d1);
			Real l2 = sqrt(r * r - d2 * d2);

			Coord3D s1 = p1.origin + l1 * dir01;
			Coord3D s2 = p2.origin + l2 * dir02;
			TSegment3D<Real> seg12 = Segment3D(v1, v2);
			//case 7
			if (center.distance(seg12) <= r)
			{
				ret += Triangle3D(v0, v1, v2).area();

				TPoint3D<Real> p0 = center.project(seg12);
				Real l = sqrt(r * r - center.distanceSquared(p0));
				
				Coord3D s10 = p0.origin - l * dir12;
				Coord3D s20 = p0.origin + l * dir12;

				ret -= TTriangle3D<Real>(v1, s1, s10).area();
				ret -= TTriangle3D<Real>(v2, s2, s20).area();

				Real d2 = center.distance(Segment3D(s2, s20));
				Real d20 = 0.5 * (s2 - s20).norm();
				Real angle2 = asin(d20 / r);
				Real secArea2 = angle2 * r * r - d2 * d20;

				Real d1 = center.distance(Segment3D(s1, s10));
				Real d10 = 0.5 * (s1 - s10).norm();
				Real angle1 = asin(d10 / r);
				Real secArea1 = angle1 * r * r - d1 * d10;

				ret += secArea1 + secArea2;
			}
			//case 6
			else
			{
				ret += TTriangle3D<Real>(v0, s1, s2).area();

				Real d0 = center.distance(Segment3D(s1, s2));
				Real d12 = 0.5 * (s1 - s2).norm();
				Real angle = asin(d12 / r);

				//check whether the center is located inside the triangle
				bool opposite = (v0 - s1).dot(center.origin - s1) < 0;

				//calculate sector area
				Real secArea = opposite ? (M_PI - angle) * r * r : angle * r * r;

				//subtract/add the triangle from/to the sector
				secArea += opposite ? d0 * d12 : -d0 * d12;

				ret += secArea;
			}

			return maximum(ret, Real(0));
		}

		//case 2, 3, 4 and 5
		Real circleArea = Real(M_PI) * r * r;
		Real ret = circleArea;
		for (int j = 0; j < 3; j++)
		{
			Coord3D v0 = triangle.v[j];
			Coord3D v1 = triangle.v[(j + 1) % 3];
			Coord3D v2 = triangle.v[(j + 2) % 3];
			TSegment3D<Real> seg12 = TSegment3D<Real>(v1, v2);

			TPoint3D<Real> p = center.project(seg12);

			Real secArea = Real(0);
			Real d0 = center.distance(p);
			if (d0 <= r)
			{
				Real d1 = sqrt(r * r - d0 * d0);
				Real angle = asin(d1 / r);

				//check whether the center is located inside the triangle
				bool opposite = (p.origin - center.origin).dot(p.origin - v0) < 0;

				//calculate sector area
				secArea = opposite ? (M_PI - angle) * r * r : angle * r * r;

				//subtract/add the triangle from/to the sector
				secArea += opposite ? d0 * d1 : -d0 * d1;
			}

			ret -= secArea;
		}
		return maximum(ret, Real(0));
	}
}
