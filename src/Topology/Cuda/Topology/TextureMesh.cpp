#include "TextureMesh.h"

namespace dyno
{
	TextureMesh::TextureMesh()
		: TopologyModule()
	{
	}

	TextureMesh::~TextureMesh()
	{
		mVertices.clear();
		mNormals.clear();
		mTexCoords.clear();

		mMaterials.clear();
		mShapes.clear();
	}

}