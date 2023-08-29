#pragma once

#include <Node.h>
#include <Topology/TriangleSet.h>

using namespace dyno;

class ObjMeshNode : public Node
{
public:
	ObjMeshNode();

	bool load(const std::string& path);

public:
	// use TriangleSet data structure?

	DEF_INSTANCE_OUT(TriangleSet<DataType3f>, TriangleSet, "");

	// additional data
	DEF_ARRAY_OUT(Vec3f, Normal, DeviceType::GPU, "");
	DEF_ARRAY_OUT(Vec3i, NormalIndex, DeviceType::GPU, "");

	DEF_ARRAY_OUT(Vec2f, TexCoord, DeviceType::GPU, "");
	DEF_ARRAY_OUT(Vec3i, TexCoordIndex, DeviceType::GPU, "");

	DEF_ARRAY2D_OUT(Vec4f, ColorTexture, DeviceType::GPU, "");
};
