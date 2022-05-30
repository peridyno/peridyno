#include "imgui_extend.h"
namespace ImGui{
ImVec4      ExColorsVal[ImGuiExColVal_COUNT];
};

void ImGui::BeginHorizontal(){
    ImGuiWindow* window = GetCurrentWindow();
    // window->DC.CursorPos = window->DC.CursorMaxPos = ImVec2(bar_rect.Min.x + window->DC.MenuBarOffset.x, bar_rect.Min.y + window->DC.MenuBarOffset.y);
    window->DC.LayoutType = ImGuiLayoutType_Horizontal;
}

void ImGui::EndHorizontal(){
    ImGuiWindow* window = GetCurrentWindow();
    window->DC.LayoutType = ImGuiLayoutType_Vertical;
}

void ImGui::sampleButton(const char* label, bool *v)
{
    float padding = 10.0f;
    float bounding = 1.0f;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 label_size = ImGui::CalcTextSize(label);
    const ImVec2 button_size = ImVec2(label_size.x + padding * 2, label_size.y + padding * 2);
    const ImVec2 bound_size =  ImVec2(button_size.x + bounding * 2, button_size.y + bounding * 2);
    ImVec2 p_button = ImVec2(p.x + bounding, p.y + bounding);
    ImVec2 p_label = ImVec2(p_button.x + padding, p_button.y + padding);

    float radius = bound_size.y * 0.30f;

    // 透明的按钮
    if (ImGui::InvisibleButton(label, bound_size))
        *v = !*v;
    ImVec4 col_bf4;
    ImGuiStyle& style = ImGui::GetStyle();

    // 颜色自定义
    if (ImGui::IsItemActivated()) col_bf4 = *v ? style.Colors[40] : style.Colors[23];
    else if (ImGui::IsItemHovered()) col_bf4 =  *v ? style.Colors[42] : style.Colors[24];
    else col_bf4 = *v ? style.Colors[41] : style.Colors[22];

    ImU32 col_bg = IM_COL32(255 * col_bf4.x, 255 * col_bf4.y, 255 * col_bf4.z, 255 * col_bf4.w);
    ImU32 col_text = IM_COL32(255, 255, 255, 255);
    ImU32 col_bound = IM_COL32(0,0,0,255);
    
    // 绘制矩形形状
    draw_list->AddRect(p, ImVec2(p.x + bound_size.x, p.y + bound_size.y), col_bound , radius);
    draw_list->AddRectFilled(p_button, ImVec2(p_button.x + button_size.x, p_button.y + button_size.y), col_bg, radius);
    draw_list->AddText(p_label, col_text, label);
}

void ImGui::toggleButton(ImTextureID texId, const char* label, bool *v)
{
    if (*v == true)
    {

        ImGui::PushID(label);
        ImGui::PushStyleColor(ImGuiCol_Button, ExColorsVal[ImGuiExColVal_Button_1]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ExColorsVal[ImGuiExColVal_ButtonHovered_1]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ExColorsVal[ImGuiExColVal_ButtonActive_1]);
        ImGui::ImageButtonWithText(texId, label);
        if (ImGui::IsItemClicked(0))
        {
            *v = !*v;
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();
    }
    else
    {
        if (ImGui::ImageButtonWithText(texId ,label))
            *v = true;
    }
}

void ImGui::toggleButton(const char* label, bool *v)
{
    if (*v == true)
    {

        ImGui::PushID(label);
        ImGui::PushStyleColor(ImGuiCol_Button, ExColorsVal[ImGuiExColVal_Button_1]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ExColorsVal[ImGuiExColVal_ButtonHovered_1]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ExColorsVal[ImGuiExColVal_ButtonActive_1]);        
        ImGui::Button(label);
        if (ImGui::IsItemClicked(0))
        {
            *v = !*v;
        }
        ImGui::PopStyleColor(3);
        ImGui::PopID();
    }
    else
    {
        if (ImGui::Button(label))
            *v = true;
    }
}
bool ImGui::ImageButtonWithText(ImTextureID texId,const char* label,const ImVec2& imageSize, const ImVec2 &uv0, const ImVec2 &uv1, int frame_padding, const ImVec4 &bg_col, const ImVec4 &tint_col) {
    ImGuiWindow* window = GetCurrentWindow();
    
    if (window->SkipItems)
    return false;

    ImVec2 size = imageSize;
    if (size.x<=0 && size.y<=0) {size.x=size.y=ImGui::GetTextLineHeightWithSpacing();}
    else {
        if (size.x<=0)          size.x=size.y;
        else if (size.y<=0)     size.y=size.x;
        size*=window->FontWindowScale*ImGui::GetIO().FontGlobalScale;
    }

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    const ImGuiID id = window->GetID(label);
    const ImVec2 textSize = ImGui::CalcTextSize(label,NULL,true);
    const bool hasText = textSize.x>0;

    const float innerSpacing = hasText ? ((frame_padding >= 0) ? (float)frame_padding : (style.ItemInnerSpacing.x)) : 0.f;
    const ImVec2 padding = (frame_padding >= 0) ? ImVec2((float)frame_padding, (float)frame_padding) : style.FramePadding;
    const ImVec2 totalSizeWithoutPadding(size.x+innerSpacing+textSize.x,size.y>textSize.y ? size.y : textSize.y);
    const ImRect bb(window->DC.CursorPos, window->DC.CursorPos + totalSizeWithoutPadding + padding*2);
    ImVec2 start(0,0);
    start = window->DC.CursorPos + padding;
    if (size.y<textSize.y) start.y+=(textSize.y-size.y)*.5f;

    const ImRect image_bb(start, start + size);
    start = window->DC.CursorPos + padding;start.x+=size.x+innerSpacing;
    if (size.y>textSize.y) start.y+=(size.y-textSize.y)*.5f;
    
    ItemSize(bb);
    if (!ItemAdd(bb, id))
    return false;

    bool hovered=false, held=false;
    bool pressed = ButtonBehavior(bb, id, &hovered, &held);

    // Render
    const ImU32 col = GetColorU32((hovered && held) ? ImGuiCol_ButtonActive : hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button);
    RenderFrame(bb.Min, bb.Max, col, true, ImClamp((float)ImMin(padding.x, padding.y), 0.0f, style.FrameRounding));
    if (bg_col.w > 0.0f)
    window->DrawList->AddRectFilled(image_bb.Min, image_bb.Max, GetColorU32(bg_col));

    window->DrawList->AddImage(texId, image_bb.Min, image_bb.Max, uv0, uv1, GetColorU32(tint_col));

    if (textSize.x>0) ImGui::RenderText(start,label);
    return pressed;
}

void ImGui::beginTitle(const char* label){
    // 避免label输出，ImGui ID压入栈中
    ImGui::PushID(label);
}

void ImGui::endTitle(){
    // ImGui ID弹栈
    ImGui::PopID();
}

template<typename T> 
std::shared_ptr<ImU32[]> ImGui::ToImU(T v, int size)
{
    auto vPtr = std::make_unique<ImU32[]>(size);
    for(int i = 0; i < size; ++ i)
		vPtr[i] = IM_COL32(v[i][0], v[i][1], v[i][2], 150);
    return vPtr;
}

ImU32 ImGui::VecToImU(const dyno::Vec3f *v)
{
    return IM_COL32((*v)[0] * 255, (*v)[1] * 255, (*v)[2] * 255, 150);
}

template<typename T> 
bool ImGui::ColorBar(const char* label, const float* values, const T col, int length)
{
	if (col == nullptr ) return false;
	ImGuiContext& g = *GImGui;
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImDrawList* draw_list = window->DrawList;
    ImGuiStyle& style = g.Style;
    ImGuiIO& io = g.IO;
    ImGuiID id = window->GetID(label);
    
    const float width = CalcItemWidth();
    g.NextItemData.ClearFlags();

    PushID(label);
    BeginGroup();

    float text_width = GetFrameHeight() * 2;
    float text_height = GetFrameHeight();
    float bars_width = GetFrameHeight();
    float bars_height = ImMax(bars_width * 1, width - 1 * (bars_width + style.ItemInnerSpacing.x)); // Saturation/Value picking box
    float offset_width = bars_width * 0.2;
    ImVec2 bar_pos = window->DC.CursorPos;
    ImVec2 real_bar_pos = bar_pos + ImVec2(0, 0.3 * text_height );
    ImRect bb(bar_pos, bar_pos + ImVec2(bars_width + offset_width + text_width, bars_height + text_height));
    int grid_count = length - 1;  
    
    // Set Item Size
    ItemSize(ImVec2(bars_width + offset_width + text_width, bars_height + text_height));
	if (!ItemAdd(bb, id))
	{
		EndGroup();
		PopID();
		return false;
	}
        
    for (int i = 0; i <= grid_count; ++i){
        if (i != grid_count)
        draw_list->AddRectFilledMultiColor(
            ImVec2(real_bar_pos.x, real_bar_pos.y + i * (bars_height / grid_count)), 
            ImVec2(real_bar_pos.x + bars_width, real_bar_pos.y + (i + 1) * (bars_height / grid_count)), 
            col[i], col[i], col[i + 1], col[i + 1]);

        if (values != nullptr){
            char buf[20];
            sprintf(buf,"%.3f", values[i]);
            draw_list->AddText(ImVec2(bar_pos.x + bars_width + offset_width,  bar_pos.y + i * (bars_height / grid_count)), IM_COL32(255,255,255,255),buf);
        }
    }
    RenderFrameBorder(ImVec2(real_bar_pos.x, real_bar_pos.y), ImVec2(real_bar_pos.x + bars_width, real_bar_pos.y + bars_height), 0.0f);
    EndGroup();
    PopID();
    return true;
}


ImU32 ImGui::ToHeatColor(const float v, const float v_min, const float v_max){
    float x = dyno::clamp((v - v_min) / (v_max - v_min), float(0), float(1));
    float r = dyno::clamp(float(-4 * abs(x - 0.75) + 2), float(0), float(1));
    float g = dyno::clamp(float(-4 * abs(x - 0.50) + 2), float(0), float(1));
    float b = dyno::clamp(float(-4 * abs(x) + 2), float(0), float(1));
    return IM_COL32(r * 255, g * 255, b* 255, 150);
}

ImU32 ImGui::ToJetColor(const float v, const float v_min, const float v_max){
    float x = dyno::clamp((v - v_min) / (v_max - v_min), float(0), float(1));
    float r = dyno::clamp(float(-4 * abs(x - 0.75) + 1.5), float(0), float(1));
    float g = dyno::clamp(float(-4 * abs(x - 0.50) + 1.5), float(0), float(1));
    float b = dyno::clamp(float(-4 * abs(x - 0.25) + 1.5), float(0), float(1));
    return IM_COL32(r * 255, g * 255, b* 255, 150);
}

void ImGui::initializeStyle(float scale)
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(scale);
    style.WindowRounding = 7.0f;
    style.ChildRounding = 7.0f;
    style.FrameRounding = 7.0f;
    style.PopupRounding = 7.0f;
	
	ImGuiIO& io = ImGui::GetIO();
	// Load a first font
	//ImFont* font = io.Fonts->AddFontDefault();
    //Default font as Qt
	
	std::string arialPath = getAssetPath() + "font/arial.ttf";
	io.Fonts->AddFontFromFileTTF(arialPath.c_str(), 13.0f);
	// IconFont
	ImFontConfig config;
	config.MergeMode = true;
	config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
	static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
	
	std::string solidPath = getAssetPath() + "font/fa-solid-900.ttf";
	io.Fonts->AddFontFromFileTTF(solidPath.c_str(), 13.0f, &config, icon_ranges);
	io.Fonts->Build();

    io.FontGlobalScale = scale;
}

void ImGui::initColorVal()
{
    ExColorsVal[ImGuiExColVal_Button_1]             = ImVec4(230/255.0, 179/255.0,   0/255.0, 105/255.0);
    ExColorsVal[ImGuiExColVal_ButtonHovered_1]      = ImVec4(230/255.0, 179/255.0,   0/255.0, 255/255.0);
    ExColorsVal[ImGuiExColVal_ButtonActive_1]       = ImVec4(255/255.0, 153/255.0,   0/255.0, 255/255.0);
    ExColorsVal[ImGuiExColVal_WindowTopBg_1]        = ImVec4(  0/255.0,   0/255.0,   0/255.0,  10/255.0);
}