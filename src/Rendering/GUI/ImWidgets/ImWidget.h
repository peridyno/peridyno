#pragma once

#include <Node.h>
#include <imgui.h>

namespace dyno
{
	class ImWidget : public Node
	{
	public:
		virtual bool initialize() = 0;
		virtual void update() = 0;
		virtual void paint() = 0;
	};
}