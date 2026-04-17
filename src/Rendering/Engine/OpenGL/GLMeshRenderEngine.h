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

		void renderMesh(const std::vector<FInstance<TextureMesh>*>& texmeshs, const std::vector<FInstance<TriangleSet<DataType3f>>*>& triangles, const RenderParams& rparams,bool renderTransparency = false);

		virtual std::string name() const override;

		void updateModuleGL()
		{
			realisticRenderModule->inTextureMesh()->getDataPtr();
			transparencyRealisticModule->inTextureMesh()->getDataPtr();
			surfaceRenderModule->inTriangleSet()->getDataPtr();
		};

		std::shared_ptr<GLSurfaceVisualModule> surfaceRenderModule = NULL;
		std::shared_ptr<GLPhotorealisticRender> realisticRenderModule = NULL;
		std::shared_ptr<GLPhotorealisticRender> transparencyRealisticModule = NULL;


	};
};
