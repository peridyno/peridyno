#include "gtest/gtest.h"
#include "Topology/Primitive3D.h"

using namespace dyno;

TEST(Point3D, distance)
{
	Sphere3D sphere(Coord3D(0), 1);

	Point3D point1(0, 2, 0);
	Point3D point2(0, 0.5, 0);

	EXPECT_EQ(point1.distance(sphere), 1);
	EXPECT_EQ(point2.distance(sphere), Real(-0.5));

	Point3D point3(0);
	AlignedBox3D abox;
	EXPECT_EQ(point3.distance(abox), Real(-1));

	Tet3D tet1;
	Point3D point4(0.1, 0.1, 0.1);
	EXPECT_EQ(point4.distance(tet1), Real(-0.1));

	Point3D point5(0.0, 0.1, 0.0);
	Capsule3D capsule1;
	EXPECT_TRUE(abs(point4.distance(capsule1) - (sqrt(Real(0.02)) - Real(1))) <= (std::numeric_limits<Real>::epsilon)());
	EXPECT_EQ(point5.distance(capsule1), Real(-0.9));

	Point3D point6(2, 2, 2);
	EXPECT_TRUE(abs(point6.distance(capsule1) - Real(2)) <= (std::numeric_limits<Real>::epsilon)());
}

TEST(Line3D, distance) {
    //arrange
    //act
    //assert
	Line3D line(Coord3D(0), Coord3D(1, 0, 0));
	Point3D point(-3, -4, 0);
    EXPECT_EQ (line.distance(point),  4);

	Line3D line2(Coord3D(0), Coord3D(0, 0, 0));
	EXPECT_EQ(line2.distance(point), 5);

	Line3D line3(Coord3D(0), Coord3D(1, 1, 0));
	EXPECT_EQ(line.distance(line3), 0);

	Line3D line4(Coord3D(0, 1, 0), Coord3D(1, 1, 0));
	EXPECT_EQ(line.distance(line4), 0);

	Ray3D ray1(Coord3D(0), Coord3D(0, 1, 0));
	EXPECT_EQ(line.distance(ray1), 0);

	Ray3D ray2(Coord3D(0, 1, 0), Coord3D(0, 1, 0));
	EXPECT_EQ(line.distance(ray2), 1);

	Ray3D ray3(Coord3D(0, 1, 1), Coord3D(0, 1, 0));
	EXPECT_EQ(line.distance(ray3), sqrt(Real(2)));

	Line3D line5(Coord3D(5), Coord3D(0, 0, 0));
	EXPECT_EQ(line2.distance(line5), sqrt(Real(75)));
	EXPECT_EQ(line.distance(line5), sqrt(Real(50)));
	EXPECT_EQ(line5.distance(line), sqrt(Real(50)));

	Point3D point2(1, 1, 1);
	Point3D point3(2, 2, 2);
	AlignedBox3D box(Coord3D(-1), Coord3D(1));
	EXPECT_EQ(point2.distance(box), sqrt(Real(0)));
	EXPECT_EQ(point3.distance(box), sqrt(Real(3)));

	//along x axis
	Line3D line_x_0(Coord3D(0, 3, 3), Coord3D(1, 0, 0));
	Line3D line_x_1(Coord3D(0, 1, 3), Coord3D(1, 0, 0));
	Line3D line_x_2(Coord3D(0, 1, 1), Coord3D(1, 0, 0));
	Line3D line_x_4(Coord3D(0, -3, -3), Coord3D(1, 0, 0));
	EXPECT_EQ(line_x_0.distance(box), sqrt(Real(8)));
	EXPECT_EQ(line_x_1.distance(box), Real(2));
	EXPECT_EQ(line_x_2.distance(box), Real(0));
	EXPECT_EQ(line_x_4.distance(box), sqrt(Real(8)));

	//along y axis
	Line3D line6(Coord3D(3, 0, 1), Coord3D(0, 1, 0));
	Line3D line7(Coord3D(-3, 0, 1), Coord3D(0, 1, 0));
	Line3D line8(Coord3D(3, 0, 3), Coord3D(0, 1, 0));
	EXPECT_EQ(line.distance(box), Real(0));
	EXPECT_EQ(line6.distance(box), Real(2));
	EXPECT_EQ(line8.distance(box), sqrt(Real(8)));

	//along z axis
	Line3D line_z_0(Coord3D(3, 1, 0), Coord3D(0, 0, 1));
	Line3D line_z_1(Coord3D(-3, 1, 0), Coord3D(0, 0, 1));
	Line3D line_z_2(Coord3D(3, 3, 0), Coord3D(0, 0, 1));
	EXPECT_EQ(line_z_0.distance(box), Real(2));
	EXPECT_EQ(line_z_1.distance(box), Real(2));
	EXPECT_EQ(line_z_2.distance(box), sqrt(Real(8)));

	Line3D line9(Coord3D(1, 1, 1), Coord3D(1, 1, 5));
	EXPECT_EQ(line9.distance(box), Real(0));

	//test distance between degenerate lines and aligned boxes
	Line3D line_deg_0(Coord3D(2, 2, 2), Coord3D(0));
	Line3D line_deg_1(Coord3D(0.5, 0.5, 0.5), Coord3D(0));
	EXPECT_EQ(line_deg_0.distanceSquared(box), Real(3));
	EXPECT_EQ(line_deg_1.distanceSquared(box), Real(0));

	Line3D line_00_0(Coord3D(2, 0, 2), Coord3D(1, 0, -1));
	Line3D line_00_1(Coord3D(-2, 0, -2), Coord3D(1, 0, -1));
	Line3D line_00_2(Coord3D(-2, 0, -2), Coord3D(2, 0, -2));
	EXPECT_EQ(line_00_0.distanceSquared(box), (Real(2)));
	EXPECT_EQ(line_00_1.distanceSquared(box), (Real(2)));
	EXPECT_EQ(line_00_2.distanceSquared(box), (Real(2)));

	Line3D line_111_1(Coord3D(2, 0, 2), Coord3D(2, 1, -2));
	EXPECT_EQ(line_111_1.distanceSquared(box), (Real(2)));

	Coord3D vec(0);
	vec.normalize();
	EXPECT_EQ(vec.norm(), (Real(0)));

	Line3D line_inter_0(Coord3D(0, 1, 0), Coord3D(2, -1, 1));
	Plane3D plane_inter_0(Coord3D(0, 0, 0), Coord3D(0, 1, 0));
	Point3D inter_0;
	line_inter_0.intersect(plane_inter_0, inter_0);
	EXPECT_EQ(inter_0.origin[0], Real(2));
	EXPECT_EQ(inter_0.origin[2], Real(1));

	Line3D line_inter_1(Coord3D(0, 1, 0), Coord3D(-2, -1, -1));
	Point3D inter_1;
	line_inter_1.intersect(plane_inter_0, inter_1);
	EXPECT_EQ(inter_1.origin[0], Real(-2));
	EXPECT_EQ(inter_1.origin[2], Real(-1));

	AlignedBox3D abox1;
	Line3D line_inter_2;
	Segment3D segTmp;
	EXPECT_EQ(line_inter_2.intersect(abox1, segTmp), 2);
	Line3D line_inter_3(Coord3D(0, 1, 0), Coord3D(1, 0, 0));
	EXPECT_EQ(line_inter_3.intersect(abox1, segTmp), 0);


	Ray3D ray_inter_1(Coord3D(-0.1, 0, 0), Coord3D(1, 0, 0));
	Ray3D ray_inter_2(Coord3D(-0.1, 0, -0.1), Coord3D(1, 0, 1));
	Ray3D ray_inter_3(Coord3D(-1.1, 0, -1.1), Coord3D(1, 0, 1));
	EXPECT_EQ(ray_inter_1.intersect(abox1, segTmp), 1);
	EXPECT_EQ(ray_inter_2.intersect(abox1, segTmp), 1);
	EXPECT_EQ(ray_inter_3.intersect(abox1, segTmp), 2);

	Segment3D seg_inter_1(Coord3D(-0.1, 0, 0), Coord3D(0.9, 0, 0));
	Segment3D seg_inter_2(Coord3D(-0.1, 0, -0.1), Coord3D(1, 0, 1));
	Segment3D seg_inter_3(Coord3D(-1.1, 0, -1.1), Coord3D(1, 0, 1));
	EXPECT_EQ(seg_inter_1.intersect(abox1, segTmp), 0);
	EXPECT_EQ(seg_inter_2.intersect(abox1, segTmp), 1);
	EXPECT_EQ(seg_inter_3.intersect(abox1, segTmp), 2);
}

TEST(Ray3D, distance) {
	//arrange
	//act
	//assert
	Ray3D ray(Coord3D(0), Coord3D(1, 0, 0));
	Point3D point(-3, -4, 0);
	EXPECT_EQ(point.distance(ray), 5);

	Ray3D ray2(Coord3D(0), Coord3D(0, 0, 0));
	EXPECT_EQ(point.distance(ray2), 5);

	Segment3D seg1(Coord3D(1, 1, 1), Coord3D(1, 1, 1));
	EXPECT_EQ(ray.distance(seg1), sqrt(Real(2)));

	Segment3D seg2(Coord3D(1, 1, 1), Coord3D(1, 1, 2));
	EXPECT_EQ(ray.distance(seg2), sqrt(Real(2)));
}

TEST(Segement3D, distance) {
	//arrange
	//act
	//assert
	Segment3D seg(Coord3D(0), Coord3D(1, 0, 0));
	Point3D point(-3, -4, 0);
	EXPECT_EQ(point.distance(seg), 5);

	Point3D point2(4, 4, 0);
	EXPECT_EQ(point2.distance(seg), 5);

	Segment3D seg2(Coord3D(0), Coord3D(0, 1, 0));
	Segment3D seg3(Coord3D(2, 0, 0), Coord3D(2, 0.5, 0));
	EXPECT_EQ(seg2.distance(seg3), 2);

	Segment3D seg4(Coord3D(0), Coord3D(0));
	EXPECT_EQ(seg4.distance(seg3), 2);

	Triangle3D tri1(Coord3D(0), Coord3D(1, 0, 0), Coord3D(0, 1, 0));
	EXPECT_EQ(seg4.distance(tri1), Real(0));

	Segment3D seg5(Coord3D(1, 1, 0), Coord3D(1, 1, 1));
	EXPECT_EQ(seg5.distance(tri1), sqrt(Real(2))/Real(2));

	Segment3D seg6(Coord3D(0.5, 0.5, 0), Coord3D(0.5, 0.5, 1));
	EXPECT_EQ(seg6.distance(tri1), Real(0));

	Segment3D seg7(Coord3D(0.5, -0.5, 0), Coord3D(0.5, -0.5, 5));
	EXPECT_EQ(seg7.distance(tri1), Real(0.5));

	Segment3D seg8(Coord3D(-0.5, 0.5, 0), Coord3D(-0.5, 0.5, 1));
	EXPECT_EQ(seg8.distance(tri1), Real(0.5));

	Segment3D seg9(Coord3D(-0.5, 0.5, -1), Coord3D(-0.5, 0.5, 1));
	EXPECT_EQ(seg8.distance(tri1), Real(0.5));
}

TEST(Triangle3D, fun) {
	Triangle3D tri(Coord3D(0), Coord3D(1, 0, 0), Coord3D(0, 1, 0));

	Point3D point1(5, 0, 0);
	Point3D point2(0, 4, 0);
	Point3D point3(-3, -4, 0);
	Point3D point4(0.5, 0.5, 10);

	EXPECT_EQ(point1.distance(tri), 4);
 	EXPECT_EQ(point2.distance(tri), 3);
 	EXPECT_EQ(point3.distance(tri), 5);
	EXPECT_EQ(point4.distance(tri), 10);

	Triangle3D tri2(Coord3D(0), Coord3D(1, 0, 0), Coord3D(0.5, 0, 0));
	EXPECT_EQ(point1.distance(tri2), 4);

	Triangle3D tri3(Coord3D(0), Coord3D(1, 0, 0), Coord3D(0, 0.5, 0));
	Real a = tri3.area();
    EXPECT_EQ(tri3.area(), 0.25);

	Triangle3D::Param param;
	Point3D point6(0, 0, 0);
	tri.computeBarycentrics(point6.origin, param);
	EXPECT_EQ(abs(param.u - 1.0) < EPSILON, true);

	Point3D point7(1, 0, 0);
	tri.computeBarycentrics(point7.origin, param);
	EXPECT_EQ(abs(param.v - 1.0) < EPSILON, true);

	Point3D point8(0, 1, 0);
	tri.computeBarycentrics(point8.origin, param);
	EXPECT_EQ(abs(param.w - 1.0) < EPSILON, true);
}

TEST(Disk3D, distance) {
	Disk3D disk(Coord3D(0), Coord3D(1, 0, 0), 1);

	Point3D point1(0, 2, 0);
	Point3D point2(0, 0.5, 0);
	Point3D point3(1, 0, 0);

	EXPECT_EQ(point1.distance(disk), 1);
	EXPECT_EQ(point2.distance(disk), 0);
	EXPECT_EQ(point3.distance(disk), 1);
}

TEST(Tet3D, function) {
	Tet3D tet;

	Point3D point1(0, 2, 0);
	Point3D point2(0, 0.5, 0);
	Point3D point3(-1, 0, 0);

	EXPECT_EQ(point1.distance(tet), 1);
	EXPECT_EQ(point2.distance(tet), 0);
	EXPECT_EQ(point3.distance(tet), 1);

	Tet3D tet1(Coord3D(0), Coord3D(1, 0, 0), Coord3D(0, 0, 1), Coord3D(0, 1, 0));
	EXPECT_FALSE(tet1.isValid());
	EXPECT_EQ(tet1.volume(), Real(-1.0f/6.0f));

	Point3D c = tet1.circumcenter();
	EXPECT_EQ((c.origin - Coord3D(0.5f)).norm() < EPSILON, true);

	Tet3D tet2(Coord3D(1.84683, -0.01341793, -2.8445360000000002), Coord3D(1.83172,  0.080868700000000002, -2.722531), Coord3D(1.8336870000000001,  0.044794059999999997, -2.8555130000000002), Coord3D(1.8130956498821573,  0.0032346277719787972, -2.844774865334295));
	c = tet2.circumcenter();
	EXPECT_EQ(c.inside(tet2), false);

	Point3D p0(0.1f, 0.1f, 0.1f);
	EXPECT_EQ(p0.inside(tet1), true);

	Tet3D tet3(Coord3D(-2.8, -2.8, -2.8), Coord3D(0, 0, 1), Coord3D(0, 1, 0), Coord3D(1, 0, 0));
	c = tet3.circumcenter();
	EXPECT_EQ(c.inside(tet3), true);

	Tet3D tet4(Coord3D(1, 0, 0), Coord3D(0), Coord3D(0, 0, 1), Coord3D(0, 1, 0));
	c = tet4.circumcenter();
	EXPECT_EQ((c.origin - Coord3D(0.5f)).norm() < EPSILON, true);

	Tet3D tet5(Coord3D(1, 0, 0), Coord3D(-0.8, -0.8, -0.8), Coord3D(0, 1, 0), Coord3D(0, 0, 1));
	c = tet5.circumcenter();
	EXPECT_EQ(c.inside(tet5), true);

	Tet3D tet6(Coord3D(1.029422, 0.793586, 0.308014), Coord3D(1.018432, 0.824708, 0.338561), Coord3D(0.994443, 0.767503, 0.318724), Coord3D(1.027653, 0.780213, 0.356668));
	c = tet6.circumcenter();
	Point3D bc = tet6.barycenter();
	Real d = bc.distance(tet6);
	Real d1 = c.distance(tet6);
	EXPECT_EQ(c.inside(tet6), false);
	EXPECT_EQ(bc.inside(tet6), true);
}

TEST(Sphere3D, distance) {
	

	
}

TEST(OrientedBox3D, distance) {
	OrientedBox3D obb;

	Point3D point1(0, 2, 0);
	Point3D point2(0, 0.5, 0);

	EXPECT_EQ(point1.distance(obb), 1);
	EXPECT_EQ(point2.distance(obb), Real(-0.5));
}

using namespace glm;

#define FLOAT_MAX 1000000

struct Box
{
	vec4 center;
	vec4 halfLength;
	vec4 rot;
};

vec3 quat_rotate(vec4 quat, vec3 v)
{
	// Extract the vector part of the quaternion
	vec3 u = vec3(quat.x, quat.y, quat.z);

	// Extract the scalar part of the quaternion
	float s = quat.w;

	// Do the math
	return    2.0f * dot(u, v) * u
		+ (s*s - dot(u, u)) * v
		+ 2.0f * s * cross(u, v);
}

bool checkCollision(Box box0, Box box1, vec3& cNormal, float& cDist, float& outRA, float& outRB)
{
	float d = FLOAT_MAX;
	vec3 normalA = vec3(0);

	vec3 v = vec3(box1.center.x, box1.center.y, box1.center.z) - vec3(box0.center.x, box0.center.y, box0.center.z);

	//Compute A's basis  
	vec3 VAx = quat_rotate(box0.rot, vec3(1, 0, 0));
	vec3 VAy = quat_rotate(box0.rot, vec3(0, 1, 0));
	vec3 VAz = quat_rotate(box0.rot, vec3(0, 0, 1));

	vec3 VA[3];
	VA[0] = VAx;
	VA[1] = VAy;
	VA[2] = VAz;

	//Compute B's basis  
	vec3 VBx = quat_rotate(box1.rot, vec3(1, 0, 0));
	vec3 VBy = quat_rotate(box1.rot, vec3(0, 1, 0));
	vec3 VBz = quat_rotate(box1.rot, vec3(0, 0, 1));

	vec3 VB[3];
	VB[0] = VBx;
	VB[1] = VBy;
	VB[2] = VBz;

	vec3 T = vec3(dot(v, VAx), dot(v, VAy), dot(v, VAz));

	float R[3][3];
	float FR[3][3];
	float ra, rb, t;

	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			R[i][k] = dot(VA[i], VB[k]);
			FR[i][k] = abs(R[i][k]);
		}
	}

	// A's basis vectors  
	for (int i = 0; i < 3; i++)
	{
		ra = box0.halfLength[i];
		rb = box1.halfLength[0] * FR[i][0] + box1.halfLength[1] * FR[i][1] + box1.halfLength[2] * FR[i][2];
		t = abs(T[i]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, VA[i]) > 0 ? VA[i] : -VA[i];
			}
		}
	}

	// B's basis vectors  
	for (int k = 0; k < 3; k++)
	{
		ra = box0.halfLength[0] * FR[0][k] + box0.halfLength[1] * FR[1][k] + box0.halfLength[2] * FR[2][k];
		rb = box1.halfLength[k];
		t = abs(T[0] * R[0][k] + T[1] * R[1][k] + T[2] * R[2][k]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, VB[k]) > 0 ? VB[k] : -VB[k];
			}
		}
	}

	//9 cross products  
	bool parallel = false;
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			if (FR[i][k] + EPSILON >= 1.0)
				parallel = true;
		}
	}

	if (!parallel)
	{
		//L = A0 x B0  
		ra = box0.halfLength[1] * FR[2][0] + box0.halfLength[2] * FR[1][0];
		rb = box1.halfLength[1] * FR[0][2] + box1.halfLength[2] * FR[0][1];
		t = abs(T[2] * R[1][0] - T[1] * R[2][0]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[0];
				vec3 e1 = VB[0];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A0 x B1  
		ra = box0.halfLength[1] * FR[2][1] + box0.halfLength[2] * FR[1][1];
		rb = box1.halfLength[0] * FR[0][2] + box1.halfLength[2] * FR[0][0];
		t = abs(T[2] * R[1][1] - T[1] * R[2][1]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[0];
				vec3 e1 = VB[1];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A0 x B2  
		ra = box0.halfLength[1] * FR[2][2] + box0.halfLength[2] * FR[1][2];
		rb = box1.halfLength[0] * FR[0][1] + box1.halfLength[1] * FR[0][0];
		t = abs(T[2] * R[1][2] - T[1] * R[2][2]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[0];
				vec3 e1 = VB[2];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A1 x B0  
		ra = box0.halfLength[0] * FR[2][0] + box0.halfLength[2] * FR[0][0];
		rb = box1.halfLength[1] * FR[1][2] + box1.halfLength[2] * FR[1][1];
		t = abs(T[0] * R[2][0] - T[2] * R[0][0]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[1];
				vec3 e1 = VB[0];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A1 x B1  
		ra = box0.halfLength[0] * FR[2][1] + box0.halfLength[2] * FR[0][1];
		rb = box1.halfLength[0] * FR[1][2] + box1.halfLength[2] * FR[1][0];
		t = abs(T[0] * R[2][1] - T[2] * R[0][1]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[1];
				vec3 e1 = VB[1];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A1 x B2  
		ra = box0.halfLength[0] * FR[2][2] + box0.halfLength[2] * FR[0][2];
		rb = box1.halfLength[0] * FR[1][1] + box1.halfLength[1] * FR[1][0];
		t = abs(T[0] * R[2][2] - T[2] * R[0][2]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[1];
				vec3 e1 = VB[2];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A2 x B0  
		ra = box0.halfLength[0] * FR[1][0] + box0.halfLength[1] * FR[0][0];
		rb = box1.halfLength[1] * FR[2][2] + box1.halfLength[2] * FR[2][1];
		t = abs(T[1] * R[0][0] - T[0] * R[1][0]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[2];
				vec3 e1 = VB[0];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A2 x B1  
		ra = box0.halfLength[0] * FR[1][1] + box0.halfLength[1] * FR[0][1];
		rb = box1.halfLength[0] * FR[2][2] + box1.halfLength[2] * FR[2][0];
		t = abs(T[1] * R[0][1] - T[0] * R[1][1]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[2];
				vec3 e1 = VB[1];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}

		//L = A2 x B2  
		ra = box0.halfLength[0] * FR[1][2] + box0.halfLength[1] * FR[0][2];
		rb = box1.halfLength[0] * FR[2][1] + box1.halfLength[1] * FR[2][0];
		t = abs(T[1] * R[0][2] - T[0] * R[1][2]);
		if (t > ra + rb)
			return false;
		else
		{
			float tmp_d = ra + rb - t;
			if (tmp_d < d)
			{
				vec3 e0 = VA[2];
				vec3 e1 = VB[2];
				vec3 dir = cross(e0, e1);

				vec3 dirN = length(dir) > EPSILON ? normalize(dir) : normalize(v - dot(v, e0)*e0);

				d = tmp_d;
				outRA = ra;
				outRB = rb;
				normalA = dot(v, dirN) > 0 ? dirN : -dirN;
			}
		}
	}

	cNormal = normalA;
	cDist = d;

	return true;
}

TEST(OrientedBox3D, intersectWithOBB) {
	OrientedBox3D obb;

	Point3D point1(0, 2, 0);
	Point3D point2(0, 0.5, 0);

	Box boxA;
	Box boxB;

	boxA.center = vec4(0.0f, 1.0f, 0.0f, 0.0f);
	boxA.halfLength = vec4(0.3f, 0.3f, 0.3f, 0.0f);
	boxA.rot = vec4(0.0f, 0.0f, 0.0f, 1.0f);

	boxB.center = vec4(0.0f, 1.5f, 0.0f, 0.0f);
	boxB.halfLength = vec4(0.3f, 0.3f, 0.3f, 0.0f);
	boxB.rot = vec4(0.0f, 0.0f, 0.0f, 1.0f);

	vec3 normal;
	float distance;
	float rA;
	float rB;
	checkCollision(boxA, boxB, normal, distance, rA, rB);

	EXPECT_EQ(point1.distance(obb), 1);
	EXPECT_EQ(point2.distance(obb), Real(-0.5));
}