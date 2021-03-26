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

		DArray<float3> vertices;
		DArray<float3> normals;
		DArray<float3> colors;





		DArray<float3> standard_sphere_position;
		DArray<int> standard_sphere_index;
		DArray<int> mapping;
		DArray<int> mapping_shape;
		DArray<int> attr;

		int x_segments = 50;
		int y_segments = 50;

		int num_triangle_sphere = 0;

		std::shared_ptr<TriangleRender> m_triangleRender;




		DArray<Sphere3D> m_spheres;
		DArray<Box3D> m_boxes;

		//for test
		DArray<float3> centre_sphere;
		DArray<float> radius_sphere;

		DArray<float3> centre_box;
		DArray<float3> u;
		DArray<float3> v;
		DArray<float3> w;
		DArray<float3> ext_box;
	};

}