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
		void setColor(Vec3f color);

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		Vec3f m_color;

		DArray<float3> vertices;
		DArray<float3> normals;
		DArray<float3> colors;

		std::shared_ptr<TriangleRender> m_triangleRender;
	};

}