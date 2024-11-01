#pragma once

#include "Node/ParametricModel.h"

#include "FilePath.h"

#include "Topology/TriangleSet.h"
#include "Topology/TextureMesh.h"

namespace dyno
{

	class TexturedMesh : public ParametricModel<DataType3f>
	{
		DECLARE_CLASS(TexturedMesh)
	public:
		TexturedMesh();
		~TexturedMesh() override;
		std::string getNodeType() override { return "IO"; }

	public:

		DEF_VAR(FilePath, FileName, "", "The full obj file name");

		DEF_INSTANCE_STATE(TextureMesh, TextureMesh, "");

	protected:
		void resetStates() override;

	private:
		void callbackLoadFile();
		void callbackTransform();

		DArray<Vec3f> mInitialVertex;
		DArray<Vec3f> mInitialNormal;
		DArray<Vec2f> mInitialTexCoord;
	};

}