#pragma once
#include "Topology/TextureMesh.h"

#include <Field/FilePath.h>

namespace dyno
{
	bool loadTextureMeshFromObj(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname,bool useToCenter = true);

	bool loadObj(std::vector<Vec3f>& points, std::vector<TopologyModule::Triangle>& triangles, std::string filename, bool append = false);
	
}
