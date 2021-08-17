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
			
			// mNum = this->inColor()->getDataPtr()->size();
			// mCol = ImGui::ToImU<dyno::DArray<dyno::Vec3f>>(this->inColor()->getData(), mNum);
			// mCol = ImGui::ToImU<const dyno::Vec3f*>(col, mNum);
			// mNum = 6 + 1;
			
			//TODO more color
			mColorBuffer.assign(this->inColor()->getData());
			auto min_color = m_reduce_vec3f.minimum(mColorBuffer.begin(), mColorBuffer.size());
			auto max_color = m_reduce_vec3f.maximum(mColorBuffer.begin(), mColorBuffer.size());
			
			mMinCol = ImGui::VecToImU(&min_color);
			mMaxCol = ImGui::VecToImU(&max_color);
		}else{
			// mNum = 1;
			// mCol = ImGui::ToImU<dyno::Vec3f*>(&mBaseColor, mNum);
			mMinCol = mMaxCol = ImGui::VecToImU(&mBaseColor);
		}
	}

	void ImColorbar::paintGL(RenderMode mode)
	{
		if(mode == RenderMode::COLOR){
			ImGui::Begin("Right sidebar", NULL, /*ImGuiWindowFlags_NoMove |*/ ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize);
			// ImGui::ColorBar("ColorBar", mValue, mCol, mNum);
			ImGui::ColorBar("ColorBar", mMinCol, mMaxCol);
			ImGui::End();
		}
	}
}
