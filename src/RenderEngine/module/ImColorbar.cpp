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
		mColor = nullptr;
		mValue = nullptr;
		return true;
	}

	void ImColorbar::updateGL()
	{
		Node* parent = getParent();

		if (parent == NULL)
			return;

		if (!parent->isVisible())
			return;

		if (!this->inColor()->isEmpty()){
			// TODO 
			// mNum = this->inColor()->getDataPtr()->size();
			// mCol = ImGui::ToImU<dyno::DArray<dyno::Vec3f>>(this->inColor()->getData(), mNum);

			mNum = 6 + 1;
			mCol = ImGui::ToImU<const dyno::Vec3f*>(col, mNum);
			std::cout << "Full" << std::endl;
		}else{
			mNum = 1;
			mCol = ImGui::ToImU<dyno::Vec3f*>(&mBaseColor, mNum);
			std::cout << "Base" << std::endl;
		}
	}

	void ImColorbar::paintGL(RenderMode mode)
	{
		if(mode == RenderMode::COLOR){
			ImGui::Begin("Right sidebar", NULL, /*ImGuiWindowFlags_NoMove |*/ ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			ImGui::ColorBar("ColorBar", mValue, mCol, mNum);
			ImGui::End();
		}
	}
}
