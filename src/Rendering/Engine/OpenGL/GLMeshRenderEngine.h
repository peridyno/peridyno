#pragma once

#include <memory>

#include "GLRenderEngine.h"

#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"
#include "GLPhotorealisticRender.h"
#include "GLSurfaceVisualModule.h"


namespace dyno
{
	class GLMeshRenderEngine : public GLRenderEngine
	{
	public:
		GLMeshRenderEngine();
		~GLMeshRenderEngine();

		void addField(FBase* field);

		virtual std::string name() const override;

		std::shared_ptr<SceneGraph> renderSceneGraph = NULL;
	};
};
