#pragma once

#include "Framework/ModuleVisual.h"
#include "PointRender.h"
#include "LineRender.h"
#include "TriangleRender.h"
#include "Topology/Primitive3D.h"

namespace dyno
{
	class ElementRender : public VisualModule
	{
		DECLARE_CLASS(ElementRender)
	public:
		typedef typename TSphere3D<Real> Sphere3D;
		typedef typename TOrientedBox3D<Real> Box3D;

		ElementRender();
		~ElementRender();

		void display() override;
		void setColor(Vector3f color);

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		Vector3f m_color;

		GArray<float3> vertices;
		GArray<float3> normals;
		GArray<float3> colors;





		GArray<float3> standard_sphere_position;
		GArray<int> standard_sphere_index;
		GArray<int> mapping;
		GArray<int> mapping_shape;
		GArray<int> attr;

		int x_segments = 50;
		int y_segments = 50;

		int num_triangle_sphere = 0;

		std::shared_ptr<TriangleRender> m_triangleRender;




		GArray<Sphere3D> m_spheres;
		GArray<Box3D> m_boxes;

		//for test
		GArray<float3> centre_sphere;
		GArray<float> radius_sphere;

		GArray<float3> centre_box;
		GArray<float3> u;
		GArray<float3> v;
		GArray<float3> w;
		GArray<float3> ext_box;
	};

}