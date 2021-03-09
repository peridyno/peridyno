#pragma once

#include "Framework/ModuleVisual.h"
#include "PointRender.h"
#include "LineRender.h"
#include "TriangleRender.h"

namespace dyno
{
	class SurfaceMeshRender : public VisualModule
	{
		DECLARE_CLASS(SurfaceMeshRender)
	public:
		SurfaceMeshRender();
		~SurfaceMeshRender();

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

		std::shared_ptr<TriangleRender> m_triangleRender;
	};

}