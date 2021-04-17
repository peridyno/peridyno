#pragma once

#include "Framework/ModuleVisual.h"
#include "PointRender.h"
#include "LineRender.h"
#include "TriangleRender.h"

namespace dyno
{
	class HeightFieldRenderModule : public VisualModule
	{
		DECLARE_CLASS(HeightFieldRenderModule)
	public:
		HeightFieldRenderModule();
		~HeightFieldRenderModule();

		enum RenderMode {
			POINT = 0,
			SPRITE,
			Instance
		};

		void display() override;
		void setRenderMode(RenderMode mode);
		void setColor(Vec3f color);

		void setColorRange(float min, float max);
		void setReferenceColor(float v);

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		RenderMode m_mode;
		Vec3f m_color;

		float m_refV;

		DArray<float3> vertices;
		DArray<float3> normals;
		DArray<float3> colors;

		DArray<glm::vec3> m_colorArray;

// 		std::shared_ptr<PointRenderUtil> point_render_util;
// 		std::shared_ptr<PointRenderTask> point_render_task;
		std::shared_ptr<PointRender> m_pointRender;
		std::shared_ptr<LineRender> m_lineRender;
		std::shared_ptr<TriangleRender> m_triangleRender;
	};

}