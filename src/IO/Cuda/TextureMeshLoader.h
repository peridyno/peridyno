#pragma once
#include "Topology/TextureMesh.h"

#include <FilePath.h>

namespace dyno
{
	bool loadTextureMeshFromObj(std::shared_ptr<TextureMesh> texMesh, const FilePath& fullname);
}
