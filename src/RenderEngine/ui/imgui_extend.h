#pragma once
#include <imgui.h>
#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif
#include <imgui_internal.h>

// dyno
#include "Vector.h"
#include "Module.h"

namespace ImGui
{
    // 水平布局
    IMGUI_API void          BeginHorizontal();
    IMGUI_API void          EndHorizontal();
    // Logo带字按钮
    IMGUI_API bool          ImageButtonWithText(ImTextureID texId,const char* label,const ImVec2& imageSize=ImVec2(0,0), const ImVec2& uv0 = ImVec2(0,0),  const ImVec2& uv1 = ImVec2(1,1), int frame_padding = -1, const ImVec4& bg_col = ImVec4(0,0,0,0), const ImVec4& tint_col = ImVec4(1,1,1,1));
    // ToggleButton (with Icon or no)
    IMGUI_API void          toggleButton(ImTextureID texId, const char* label, bool *v);
    IMGUI_API void          toggleButton(const char* label, bool *v);
    // SampleButton 可自定义形状按钮
    IMGUI_API void          sampleButton(const char* label, bool *v);
    // ColorBar 
	IMGUI_API bool          ColorBar(const char* label, const int* values, std::shared_ptr<ImU32[]> col, int length);
    IMGUI_API bool          ColorBar(const char* label, const ImU32 min_col, const ImU32 max_col);
    IMGUI_API bool          ColorBar(const char* label, const ImU32 min_col, const ImU32 max_col, int length);

    // Get ID without label showing
    IMGUI_API void          beginTitle(const char* label);
    IMGUI_API void          endTitle();

    IMGUI_API ImU32         VecToImU(const dyno::Vec3f* v);

    template<typename T> 
    std::shared_ptr<ImU32[]>  ToImU(T v, int size);
    template 
    std::shared_ptr<ImU32[]>  ToImU<dyno::Vec3f*>(dyno::Vec3f* v, int size);
    template 
    std::shared_ptr<ImU32[]>  ToImU<const dyno::Vec3f*>(const dyno::Vec3f* v, int size);
    template 
    std::shared_ptr<ImU32[]>  ToImU<dyno::DArray<dyno::Vec3f>>(dyno::DArray<dyno::Vec3f> v, int size);
}