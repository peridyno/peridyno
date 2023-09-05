#pragma once

#include "Node/ParametricModel.h"

#include "FilePath.h"

#include "Topology/TriangleSet.h"

using namespace dyno;

class ObjMeshNode : public ParametricModel<DataType3f>
{
public:
	ObjMeshNode();

public:
	// use TriangleSet data structure?

	DEF_VAR(FilePath, FileName, "", "The full obj file name");

	DEF_INSTANCE_STATE(TriangleSet<DataType3f>, TriangleSet, "");

	// additional data
	DEF_ARRAY_STATE(Vec3f, Normal, DeviceType::GPU, "");

	DEF_ARRAY_STATE(Vec2f, TexCoord, DeviceType::GPU, "");

	DEF_ARRAY_STATE(TopologyModule::Triangle, NormalIndex, DeviceType::GPU, "");
	
	DEF_ARRAY_STATE(TopologyModule::Triangle, TexCoordIndex, DeviceType::GPU, "");

	DEF_ARRAY2D_STATE(Vec4f, ColorTexture, DeviceType::GPU, "");

protected:
	void resetStates() override;
};
