/**
 * Copyright 2025 Yuzhong Guo, Ruikai Liang
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
 */

 #pragma once
 #include "BasicShape.h"
#include "Topology/TriangleSet.h"
#include "Topology/PolygonSet.h"


namespace dyno
{
    template<typename TDataType>
    class MedialConeModel : public BasicShape<TDataType>
    {
        DECLARE_TCLASS(MedialConeModel, TDataType);

    public:
        typedef typename TDataType::Real Real;
        typedef typename TDataType::Coord Coord;

        MedialConeModel();

        std::string caption() override { return "Medial Cone"; }

        BasicShapeType getShapeType() override { return BasicShapeType::MEDIALCONE; }

        NBoundingBox boundingBox() override;

    public:
        DEF_VAR(Coord, PointA, Coord(0, 0, -1), "Point A");

        DEF_VAR(Coord, PointB, Coord(0.866, 0, 0.500), "Point B");

        DEF_VAR(float, RadiusA, 0.2, "Radius A");

        DEF_VAR(float, RadiusB, 0.4, "Radius B");

        DEF_VAR(bool, TestBool, true, "");

        DEF_INSTANCE_STATE(PolygonSet<TDataType>, PolygonSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, CenterLine, "");

        DEF_VAR_OUT(TMedialCone3D<Real>, MedialCone, "");

    public:
        static void pushback_medialslab(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles, Vec3f pA, Vec3f pB, Real rA, Real rB, uint resolution = 40);

    protected:
        void resetStates() override;

    private:
        void varChanged();

        static double compute_angle(double r1, double r2, const Vec3f& c21);

        static Vec3f plane_line_intersection(const Vec3f& n, const Vec3f& p, const Vec3f& d, const Vec3f& a);

        static Mat4f rotate_mat(const Vec3f& point, const Vec3f& axis, double angle);

        static Vec3f mat_vec_mult(const Mat4f& mat, const Vec3f& vec);

        static std::vector<Vec3f> intersect_point_of_cones(const Vec3f& v1, double r1, const Vec3f& v2, double r2, const Vec3f& v3, double r3, const Vec3f& norm);

        static int generate_slab(
            const Vec3f& v1,
            double r1,
            const Vec3f& v2,
            double r2,
            const Vec3f& v3,
            double r3,
            std::vector<Vec3f>& slab_verts,
            std::vector<TopologyModule::Triangle>& slab_faces,
            double threshold = 1e-4);

        static void generate_conical_surface(
            const Vec3f& v1,
            float r1,
            const Vec3f& v2,
            float r2,
            int resolution,
            std::vector<Vec3f>& cone_verts,
            std::vector<TopologyModule::Triangle>& cone_faces);

    };
    IMPLEMENT_TCLASS(MedialConeModel, TDataType);
}
 