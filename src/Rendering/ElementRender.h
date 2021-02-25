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

		DeviceArray<float3> vertices;
		DeviceArray<float3> normals;
		DeviceArray<float3> colors;





		DeviceArray<float3> standard_sphere_position;
		DeviceArray<int> standard_sphere_index;
		DeviceArray<int> mapping;
		DeviceArray<int> mapping_shape;
		DeviceArray<int> attr;

		int x_segments = 50;
		int y_segments = 50;

		int num_triangle_sphere = 0;

		std::shared_ptr<TriangleRender> m_triangleRender;




		DeviceArray<Sphere3D> m_spheres;
		DeviceArray<Box3D> m_boxes;

		//for test
		DeviceArray<float3> centre_sphere;
		DeviceArray<float> radius_sphere;

		DeviceArray<float3> centre_box;
		DeviceArray<float3> u;
		DeviceArray<float3> v;
		DeviceArray<float3> w;
		DeviceArray<float3> ext_box;
	};

}