#include "ImColorbar.h"
// opengl
#include <glad/glad.h>
#include "RenderEngine.h"
#include "RenderTools.h"
#include "Utility.h"

#include <cuda_gl_interop.h>

// framework
#include <Topology/TriangleSet.h>
#include <Node.h>

namespace dyno
{
	IMPLEMENT_CLASS(ImColorbar)

	ImColorbar::ImColorbar()
	{
		// mPointSize = 0.001f;
		// mNumPoints = 1;
		mNum = 0;
		mCoord = ImVec2(0,0);
		this->setName("im_colorbar");
	}

	ImColorbar::~ImColorbar()
	{
		// mColorBuffer.clear();
	}

	void ImColorbar::setCoord(ImVec2 coord)
	{
		mCoord = coord;
	}

	ImVec2 ImColorbar::getCoord() const
	{
		return mCoord;
	}

	bool ImColorbar::initializeGL()
	{

		return true;
	}


	void ImColorbar::updateGL()
	{
		Node* parent = getParent();

		if (parent == NULL)
			return;

		if (!parent->isVisible())
			return;

		// TODO: get the right Module
		auto pColorSet = parent->getTopologyModule();
		if (pColorSet == nullptr)
		{
			return;
		}

		//TODO
		// auto& xyz = pPointSet->getPoints();
		// mNumPoints = xyz.size();
		mNum = 6 + 1;

		//TODO
		// mColor = this->inColor()->getDataPtr();
		// mValue = this->inValue()->getDataPtr();
		const int val[6 + 1] = {0,1,2,3,4,5,6};
		const Vec3f col[6 + 1] = { Vec3f(255,0,0), Vec3f(255,255,0), Vec3f(0,255,0), Vec3f(0,255,255), Vec3f(0,0,255), Vec3f(255,0,255), Vec3f(255,0,0) };
		mColor = col;
		mValue = val;
	}

	void ImColorbar::paintGL(RenderMode mode)
	{
		// FIXME RenderEngine make ImGui in Glfw
		std::cout << "ColorBar" << std::endl;

		ImGui::Begin("Demo");
		ImGui::Button("1");
		ImGui::Button("2");
		ImGui::Button("3");
		// ImGui::Begin("Right sidebar", NULL, /*ImGuiWindowFlags_NoMove |*/ ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
		// ImGui::ColorBar("ColorBar", mValue, mColor, mNum);
		// // setCoord(ImGui::GetWindowPos());
		// // ImGui::SetWindowPos(getCoord());
		// ImGui::SetWindowPos(ImVec2(0, 0));
		ImGui::End();
	}
}
