#pragma once

#include "Node/ParametricModel.h"

#include "FilePath.h"

#include "Topology/TriangleSet.h"

#include "gl/Shape.h"
#include "gl/Material.h"

using namespace dyno;

class ObjMeshNode : public ParametricModel<DataType3f>
{
public:
	ObjMeshNode();

public:

	DEF_VAR(FilePath, FileName, "", "The full obj file name");

	// additional data
	DEF_ARRAY_STATE(Vec3f, Vertex, DeviceType::GPU, "");
	DEF_ARRAY_STATE(Vec3f, Normal, DeviceType::GPU, "");
	DEF_ARRAY_STATE(Vec2f, TexCoord, DeviceType::GPU, "");

	DEF_INSTANCES_STATE(gl::Shape, Shape, "");
	DEF_INSTANCES_STATE(gl::Material, Material, "");

protected:
	void resetStates() override;
};
