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