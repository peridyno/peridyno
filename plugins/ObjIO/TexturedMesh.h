#pragma once

#include "Node/ParametricModel.h"

#include "FilePath.h"

#include "Topology/TriangleSet.h"

#include "GraphicsObject/Shape.h"
#include "GraphicsObject/Material.h"

namespace dyno
{

	class TexturedMesh : public ParametricModel<DataType3f>
	{
		DECLARE_CLASS(TexturedMesh)
	public:
		TexturedMesh();
		~TexturedMesh() override;

	public:

		DEF_VAR(FilePath, FileName, "", "The full obj file name");

		// additional data
		DEF_ARRAY_STATE(Vec3f, Vertex, DeviceType::GPU, "");
		DEF_ARRAY_STATE(Vec3f, Normal, DeviceType::GPU, "");
		DEF_ARRAY_STATE(Vec2f, TexCoord, DeviceType::GPU, "");

		DEF_INSTANCES_STATE(Shape, Shape, "");
		DEF_INSTANCES_STATE(Material, Material, "");

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