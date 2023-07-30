#pragma once

#include <Node.h>

using namespace dyno;

class ObjMeshNode : public Node
{
public:
	ObjMeshNode();

	bool load(const std::string& path);

public:
	// use TriangleSet data structure?
	DEF_ARRAY_OUT(Vec3f, Position,	DeviceType::GPU, "");
	DEF_ARRAY_OUT(Vec3f, Normal,	DeviceType::GPU, "");
	DEF_ARRAY_OUT(Vec2f, TexCoord,	DeviceType::GPU, "");
	DEF_ARRAY_OUT(Vec3i, Index,		DeviceType::GPU, "");

	DEF_ARRAY2D_OUT(Vec4f, TexColor, DeviceType::GPU, "");
};
