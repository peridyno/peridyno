#include "MedialConeModel.h"

#include "Primitive/Primitive3D.h"

#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"
#include "GLPointVisualModule.h"

#include "Mapping/Extract.h"
#include "Quat.h"
#include "Topology/EdgeSet.h"
#include <memory>
#include "SphereModel.h"


namespace dyno
{
    template<typename TDataType>
    MedialConeModel<TDataType>::MedialConeModel()
        : BasicShape<TDataType>()
    {
        this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->statePolygonSet()->setDataPtr(std::make_shared<PolygonSet<TDataType>>());

		this->stateCenterLine()->setDataPtr(std::make_shared<EdgeSet<TDataType>>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&MedialConeModel<TDataType>::varChanged, this));


		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);
		this->varRadiusA()->attach(callback);
		this->varRadiusB()->attach(callback);
		this->varPointA()->attach(callback);
		this->varPointB()->attach(callback);
		this->varRadiusA()->setRange(0.1, 1.0);
		this->varRadiusB()->setRange(0.1, 1.0);

        auto tsRender = std::make_shared<GLSurfaceVisualModule>();
		tsRender->setVisible(true);
		tsRender->setAlpha(0.2);
		this->stateTriangleSet()->connect(tsRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(tsRender);

		auto exES = std::make_shared<ExtractEdgeSetFromPolygonSet<TDataType>>();
		this->statePolygonSet()->connect(exES->inPolygonSet());
		this->graphicsPipeline()->pushModule(exES);

		auto clRender = std::make_shared<GLWireframeVisualModule>();
		clRender->varBaseColor()->setValue(Color(46.f / 255.f, 107.f / 255.f, 202.f / 255.f));
		clRender->varLineWidth()->setValue(1.f);
		clRender->varRenderMode()->setCurrentKey(GLWireframeVisualModule::CYLINDER);

		this->stateCenterLine()->connect(clRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(clRender);

		auto ptRender = std::make_shared<GLPointVisualModule>();
		ptRender->varBaseColor()->setValue(Color::Red());
		ptRender->varPointSize()->setValue(0.002);
		this->stateTriangleSet()->connect(ptRender->inPointSet());
		this->graphicsPipeline()->pushModule(ptRender);


		auto esRender = std::make_shared<GLWireframeVisualModule>();
		esRender->varBaseColor()->setValue(Color(0, 0, 0));
		exES->outEdgeSet()->connect(esRender->inEdgeSet());
		this->graphicsPipeline()->pushModule(esRender);

		this->stateTriangleSet()->promoteOuput();
    }

    template<typename TDataType>
    void MedialConeModel<TDataType>::resetStates()
    {
        varChanged();
    }

    template<typename TDataType>
	NBoundingBox MedialConeModel<TDataType>::boundingBox()
	{
		auto center = this->varLocation()->getData();
		auto rot = this->varRotation()->getData();
		auto scale = this->varScale()->getData();

		float radius = 0.5;

		Coord length(radius);
		length[0] *= scale[0];
		length[1] *= scale[1];
		length[2] *= scale[2];

		Quat<Real> q = this->computeQuaternion();

		q.normalize();

		TOrientedBox3D<Real> box;
		box.center = center;
		box.u = q.rotate(Coord(1, 0, 0));
		box.v = q.rotate(Coord(0, 1, 0));
		box.w = q.rotate(Coord(0, 0, 1));
		box.extent = length;

		auto AABB = box.aabb();
		auto& v0 = AABB.v0;
		auto& v1 = AABB.v1;


		NBoundingBox ret;
		ret.lower = Vec3f(v0.x, v0.y, v0.z);
		ret.upper = Vec3f(v1.x, v1.y, v1.z);

		return ret;
	}

    template<typename TDataType>
    void MedialConeModel<TDataType>::varChanged()
    {
        auto center = this->varLocation()->getData();
        auto scale = this->varScale()->getData();

        Quat<Real> q = this->computeQuaternion();

        Vec3f pA = this->varPointA()->getData();
        Vec3f pB = this->varPointB()->getData();
        Real rA = this->varRadiusA()->getData();
        Real rB = this->varRadiusB()->getData();

        Vec3f pA_world = center + q.rotate(pA * scale);
        Vec3f pB_world = center + q.rotate(pB * scale);

        TMedialCone3D<Real> medialConePrim = TMedialCone3D<Real>(pA_world, pB_world, rA, rB);

        this->outMedialCone()->setValue(medialConePrim);

        std::vector<Vec3f> vertices;
		std::vector<TopologyModule::Triangle> triangles;

		pushback_medialslab(vertices, triangles, pA_world, pB_world, rA, rB);

		this->stateTriangleSet()->getDataPtr()->setPoints(vertices);
		this->stateTriangleSet()->getDataPtr()->setTriangles(triangles);

		this->stateTriangleSet()->getDataPtr()->update();
    }

    template<typename TDataType>
	double MedialConeModel<TDataType>::compute_angle(double r1, double r2, const Vec3f& c21) {
		double r21_2 = std::max(pow(r1 - r2, 2.0), 0.0);
		if (r21_2 == 0) {
			return M_PI / 2;
		}
		double phi = std::atan(std::sqrt(std::max(dot(c21, c21) - r21_2, 0.0) / r21_2));
		return (r1 < r2) ? M_PI - phi : phi;
	}

    template<typename TDataType>
	Vec3f MedialConeModel<TDataType>::plane_line_intersection(const Vec3f& n, const Vec3f& p,
		const Vec3f& d, const Vec3f& a) {
		double vpt = dot(d, n);
		if (abs(vpt) < 1e-10) {
			std::cout << "SLAB GENERATION::Parallel Error" << std::endl;
			return { 0, 0, 0 };
		}
		double t = (dot(p, n) - dot(a, n)) / vpt;
		return Vec3f(a[0] + t * d[0], a[1] + t * d[1], a[2] + t * d[2]);
	}

    template<typename TDataType>
	Mat4f MedialConeModel<TDataType>::rotate_mat(const Vec3f& point, const Vec3f& axis, double angle) {

		double u = axis[0], v = axis[1], w = axis[2];
		double a = point[0], b = point[1], c = point[2];

		double cos_angle = cos(angle);
		double sin_angle = sin(angle);

		Mat4f m;

		m(0, 0) = u * u + (v * v + w * w) * cos_angle;
		m(0, 1) = u * v * (1 - cos_angle) - w * sin_angle;
		m(0, 2) = u * w * (1 - cos_angle) + v * sin_angle;

		m(1, 0) = u * v * (1 - cos_angle) + w * sin_angle;
		m(1, 1) = v * v + (u * u + w * w) * cos_angle;
		m(1, 2) = v * w * (1 - cos_angle) - u * sin_angle;

		m(2, 0) = u * w * (1 - cos_angle) - v * sin_angle;
		m(2, 1) = v * w * (1 - cos_angle) + u * sin_angle;
		m(2, 2) = w * w + (u * u + v * v) * cos_angle;

		m(0, 3) = (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - cos_angle) + (b * w - c * v) * sin_angle;
		m(1, 3) = (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - cos_angle) + (c * u - a * w) * sin_angle;
		m(2, 3) = (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - cos_angle) + (a * v - b * u) * sin_angle;

		m(3, 0) = 0; m(3, 1) = 0; m(3, 2) = 0; m(3, 3) = 1;

		return m;
	}

    template<typename TDataType>
	Vec3f MedialConeModel<TDataType>::mat_vec_mult(const Mat4f& mat,
		const Vec3f& vec)
	{
		Vec4f vec4 = mat * Vec4f(vec[0], vec[1], vec[2], 0);

		return Vec3f(vec4[0], vec4[1], vec4[2]);
	}

    template<typename TDataType>
	std::vector<Vec3f> MedialConeModel<TDataType>::intersect_point_of_cones(const Vec3f& v1, double r1,
		const Vec3f& v2, double r2,
		const Vec3f& v3, double r3,
		const Vec3f& norm) {

		if (r1 < 1e-3) return { v1, v1 };

		Vec3f v12 = v2 - v1;
		double phi_12 = compute_angle(r1, r2, v12);

		Vec3f v13 = v3 - v1;
		double phi_13 = compute_angle(r1, r3, v13);

		Vec3f p12 = v1 + Vec3f(v12).normalize() * cos(phi_12) * r1;
		Vec3f p13 = v1 + Vec3f(v13).normalize() * cos(phi_13) * r1;

		auto rot_mat = rotate_mat(p12, v12, M_PI / 2);
		Vec3f dir_12 = mat_vec_mult(rot_mat, norm);

		Vec3f intersect_p = plane_line_intersection(v13, p13, dir_12, p12);
		Vec3f v1p = intersect_p - v1;
		Vec3f scale = std::sqrt(std::max(r1 * r1 - dot(v1p, v1p), 1e-5)) * norm;

		return { Vec3f(intersect_p + scale),
				Vec3f(intersect_p - scale) };
	}

    template<typename TDataType>
	int MedialConeModel<TDataType>::generate_slab(
		const Vec3f& v1,
		double r1,
		const Vec3f& v2,
		double r2,
		const Vec3f& v3,
		double r3,
		std::vector<Vec3f>& slab_verts,
		std::vector<TopologyModule::Triangle>& slab_faces,
		double threshold)
	{

		Vec3f v12 = v1 - v2;
		Vec3f v13 = v1 - v3;
		Vec3f n = Vec3f(cross(v12, v13)).normalize();

		auto tangent_p1 = intersect_point_of_cones(v1, r1, v2, r2, v3, r3, n);
		double d2v1 = (tangent_p1[0] - v1).norm() - r1;

		auto tangent_p2 = intersect_point_of_cones(v2, r2, v1, r1, v3, r3, n);
		double d2v2 = (tangent_p2[0] - v2).norm() - r2;

		auto tangent_p3 = intersect_point_of_cones(v3, r3, v1, r1, v2, r2, n);
		double d2v3 = (tangent_p3[0] - v3).norm() - r3;

		if (d2v1 > threshold || d2v2 > threshold || d2v3 > threshold)
			return -1;

		slab_verts.insert(slab_verts.end(), {
			tangent_p1[0], tangent_p2[0], tangent_p3[0],
			tangent_p1[1], tangent_p2[1], tangent_p3[1]
			});

		size_t vcount = slab_verts.size();

		slab_faces.push_back(TopologyModule::Triangle(vcount - 1, vcount - 3, vcount - 2));
		slab_faces.push_back(TopologyModule::Triangle(vcount - 4, vcount - 5, vcount - 6));
		return 1;
	}

	template<typename TDataType>
	void MedialConeModel<TDataType>::generate_conical_surface(
		const Vec3f& v1,
		float r1,
		const Vec3f& v2,
		float r2,
		int resolution,
		std::vector<Vec3f>& cone_verts,
		std::vector<TopologyModule::Triangle>& cone_faces)
	{
		if (r1 < 1e-3f && r2 < 1e-3f)
			return;

		Vec3f c12 = Vec3f(v2 - v1);
		float phi = compute_angle(r1, r2, c12);
		c12.normalize();

		Vec3f start_dir = Vec3f(0.0f, 1.0f, 0.0f);
		if (std::abs(c12.dot(start_dir)) > 0.999f)
			start_dir = Vec3f(1.0f, 0.0f, 0.0f);
		start_dir = Vec3f(c12.cross(start_dir)).normalize();

		size_t start_idx = cone_verts.size();
		int local_vcount = 0;

		Mat4f rot_mat = rotate_mat(v1, c12, 2 * M_PI / resolution);

		for (int i = 0; i < resolution; ++i) {
			float cosV = std::cos(phi);
			float sinV = std::sin(phi);

			Vec3f pos;
			if (r1 < 1e-3f)
				pos = v1;
			else
				pos = v1 + (c12 * cosV + start_dir * sinV) * r1;

			cone_verts.push_back(pos);

			if (r2 < 1e-3f)
				pos = v2;
			else
				pos = v2 + (c12 * cosV + start_dir * sinV) * r2;

			cone_verts.push_back(pos);
			local_vcount = local_vcount + 2;

			start_dir = mat_vec_mult(rot_mat, start_dir);
		}

		for (int i = 0; i < local_vcount - 2; i += 2) {
			cone_faces.emplace_back(TopologyModule::Triangle(start_idx + i, start_idx + i + 3, start_idx + i + 1));
			cone_faces.emplace_back(TopologyModule::Triangle(start_idx + i, start_idx + i + 2, start_idx + i + 3));
		}

		cone_faces.emplace_back(TopologyModule::Triangle(start_idx + local_vcount - 2, start_idx + 1, start_idx + local_vcount - 1));
		cone_faces.emplace_back(TopologyModule::Triangle(start_idx + local_vcount - 2, start_idx, start_idx + 1));
	}

	template<typename TDataType>
	void MedialConeModel<TDataType>::pushback_medialslab(std::vector<Vec3f>& vertices, std::vector<TopologyModule::Triangle>& triangles, Vec3f pA, Vec3f pB, Real rA, Real rB, uint resolution)
	{

		generate_conical_surface(pA, rA, pB, rB, resolution, vertices, triangles);



		for (size_t i = 0; i < 2; i++)
		{
			Real r = i == 0 ? rA : rB;
			Vec3f p = i == 0 ? pA : pB;

			SphereModel<DataType3f>::generateStandardSphere(vertices, triangles, r, p, resolution / 2, resolution / 2);
		}
	}


	DEFINE_CLASS(MedialConeModel);
}