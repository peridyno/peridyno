/**
 * Copyright 2022 Shusen Liu
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
 *
 * Revision history:
 *
 * 2024-02-03: replace TriangleSet with PolygonSet as the major state;
 */

 #pragma once
 #include "BasicShape.h"
 
 #include "Topology/TriangleSet.h"
 #include "Topology/PolygonSet.h"
 #include "STL/Map.h"
 
 namespace dyno
 {
	 template<typename TDataType>
	 class SphereModel : public BasicShape<TDataType>
	 {
		 DECLARE_TCLASS(SphereModel, TDataType);
 
	 public:
		 typedef typename TDataType::Real Real;
		 typedef typename TDataType::Coord Coord;
 
 
 
		 SphereModel();
 
		 std::string caption() override { return "Sphere"; }
 
		 BasicShapeType getShapeType() override { return BasicShapeType::SPHERE; }
 
		 NBoundingBox boundingBox() override;
 
	 public:
		 DEF_VAR(Coord, Center, 0, "Sphere center");
 
		 DEF_VAR(Real, Radius, 0.5, "Sphere radius");
 
		 DECLARE_ENUM(SphereType,
			 Standard = 0,
			 Icosahedron = 1);
 
		 DEF_ENUM(SphereType, Type, SphereType::Standard, "Sphere type");
 
		 DEF_VAR(uint, Latitude, 32, "Latitude");
 
		 DEF_VAR(uint, Longitude, 32, "Longitude");
 
		 DEF_VAR(uint, IcosahedronStep, 1, "Step");
 
		 DEF_INSTANCE_STATE(PolygonSet<TDataType>, PolygonSet, "");
 
		 DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");
 
		 DEF_VAR_OUT(TSphere3D<Real>, Sphere, "");
 
	 public:
 
		 static void generateIcosahedron(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles, Real sphereRadius, uint step, Vec3f offset = Vec3f(0));
 
		 static void generateStandardSphere(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles, Real radius, Vec3f offset = Vec3f(0), uint latitude = 24, uint longitude = 24)
		 {
			 uint offsetIndex = vertices.size() + 1;
 
			 Real deltaTheta = M_PI / latitude;
			 Real deltaPhi = 2 * M_PI / longitude;
 
			 //Setup vertices
			 vertices.push_back(Coord(0, radius, 0) + offset);
 
			 Real theta = 0;
			 for (uint i = 0; i < latitude - 1; i++)
			 {
				 theta += deltaTheta;
 
				 Real phi = 0;
				 for (uint j = 0; j < longitude; j++)
				 {
					 phi += deltaPhi;
 
					 Real y = radius * std::cos(theta);
					 Real x = (std::sin(theta) * radius) * std::sin(phi);
					 Real z = (std::sin(theta) * radius) * std::cos(phi);
 
					 vertices.push_back(Coord(x, y, z) + offset);
				 }
			 }
 
			 vertices.push_back(Coord(0, -radius, 0) + offset);
 
			 //Setup polygon indices
			 uint numOfPolygon = latitude * longitude;
 
			 CArray<uint> counter(numOfPolygon);
 
			 uint incre = 0;
			 for (uint j = 0; j < longitude; j++)
			 {
				 counter[incre] = 3;
				 incre++;
			 }
 
			 for (uint i = 0; i < latitude - 2; i++)
			 {
				 for (uint j = 0; j < longitude; j++)
				 {
					 counter[incre] = 4;
					 incre++;
				 }
			 }
 
			 for (uint j = 0; j < longitude; j++)
			 {
				 counter[incre] = 3;
				 incre++;
			 }
 
 
 
			 for (uint j = 0; j < longitude; j++)
			 {
				 triangles.push_back(TopologyModule::Triangle(offsetIndex - 1, offsetIndex + j, offsetIndex + (j + 1) % longitude));
			 }
 
			 for (uint i = 0; i < latitude - 2; i++)
			 {
				 for (uint j = 0; j < longitude; j++)
				 {
					 triangles.push_back(TopologyModule::Triangle(offsetIndex + j, offsetIndex + j + longitude, offsetIndex + (j + 1) % longitude + longitude));
					 triangles.push_back(TopologyModule::Triangle(offsetIndex + j, offsetIndex + (j + 1) % longitude + longitude, offsetIndex + (j + 1) % longitude));
				 }
				 offsetIndex += longitude;
			 }
 
			 for (uint j = 0; j < longitude; j++)
			 {
				 triangles.push_back(TopologyModule::Triangle(offsetIndex + j, vertices.size() - 1, offsetIndex + (j + 1) % longitude));
			 }
		 }
 
	 protected:
		 void resetStates() override;
 
	 private:
		 void varChanged();
 
		 void standardSphere();
 
 
 
		 void icosahedronSphere();
 
	 };
 
 
	 IMPLEMENT_TCLASS(SphereModel, TDataType);
 }