#pragma once

#include <imgui.h>

#include <Module/VisualModule.h>

namespace dyno
{
	class ImWidget : public VisualModule
	{
	public:
		ImWidget();
		virtual ~ImWidget();

	public:
		virtual bool initialize() = 0;
		virtual void update() = 0;
		virtual void paint() = 0;

	protected:
		void updateGraphicsContext() override;
	};
}