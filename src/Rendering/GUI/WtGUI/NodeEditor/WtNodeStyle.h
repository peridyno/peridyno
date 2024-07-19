#pragma once

#include <Wt/WColor.h>
#include <Wt/WLogger.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Platform.h"
#include "json.hpp"

class Style
{
public:
	virtual ~Style() = default;

private:
	virtual void loadJsonText(std::string jsonText) = 0;

	virtual void loadJsonFile(std::string fileName) = 0;

	virtual void loadJsonFromByteArray(std::string const& jsonData) = 0;
};

class WtNodeStyle : public Style
{
public:
	WtNodeStyle();

	WtNodeStyle(std::string jsonText);

public:
	static void setNodeStyle(std::string jsonText);

private:
	void loadJsonText(std::string jsonText) override;

	void loadJsonFile(std::string fileName) override;

	void loadJsonFromByteArray(std::string const& jsonData) override;

public:
	Wt::WColor NormalBoundaryColor;
	Wt::WColor SelectedBoundaryColor;
	Wt::WColor GradientColor0;
	Wt::WColor GradientColor1;
	Wt::WColor GradientColor2;
	Wt::WColor GradientColor3;
	Wt::WColor ShadowColor;
	Wt::WColor FontColor;
	Wt::WColor FontColorFaded;

	Wt::WColor HotKeyColor0;
	Wt::WColor HotKeyColor1;
	Wt::WColor HotKeyColor2;

	Wt::WColor ConnectionPointColor;
	Wt::WColor FilledConnectionPointColor;

	Wt::WColor WarningColor;
	Wt::WColor ErrorColor;

	float PenWidth;
	float HoveredPenWidth;

	float ConnectionPointDiameter;

	float Opacity;
};

class WtConnectionStyle : public Style
{
public:
	WtConnectionStyle();

	WtConnectionStyle(std::string jsonText);

public:
	static void setConnectionStyle(std::string jsonText);

	Wt::WColor constructionColor() const;

	Wt::WColor normalColor() const;
	//Wt::WColor normalColor(std::string typeId) const;

	Wt::WColor selectedColor() const;

	Wt::WColor selectedHaloColor() const;

	Wt::WColor hoveredColor() const;

	float lineWidth() const;

	float constructionLineWidth() const;

	float pointDiameter() const;

	bool useDataDefinedColors() const;

private:
	void loadJsonText(std::string jsonText) override;

	void loadJsonFile(std::string fileName) override;

	void loadJsonFromByteArray(std::string const& jsonData) override;

private:

	Wt::WColor ConstructionColor;
	Wt::WColor NormalColor;
	Wt::WColor SelectedColor;
	Wt::WColor SelectedHaloColor;
	Wt::WColor HoveredColor;

	float LineWidth;
	float ConstructionLineWidth;
	float PointDiameter;

	bool UseDataDefinedColors;
};

class WtFlowViewStyle : public Style
{
public:
	WtFlowViewStyle();

	WtFlowViewStyle(std::string jsonText);

public:
	static void setStyle(std::string jsonText);

private:
	void loadJsonText(std::string jsonText) override;

	void loadJsonFile(std::string fileName) override;

	void loadJsonFromByteArray(std::string const& jsonData) override;

public:
	Wt::WColor BackgroundColor;
	Wt::WColor FineGridColor;
	Wt::WColor CoarseGridColor;
};

class WtStyleCollection
{
public:
	static WtNodeStyle const& nodeStyle();

	static WtConnectionStyle const& connectionStyle();

	static WtFlowViewStyle const& flowViewStyle();

public:
	static void setNodeStyle(WtNodeStyle);

	static void setConnectionStyle(WtConnectionStyle);

	static void setFlowViewStyle(WtFlowViewStyle);

private:
	WtStyleCollection() = default;

	WtStyleCollection(WtStyleCollection const&) = delete;

	WtStyleCollection& operator=(WtStyleCollection const&) = delete;

	static WtStyleCollection& instance();

private:
	WtNodeStyle _nodeStyle;
	WtConnectionStyle _connectionStyle;
	WtFlowViewStyle _flowViewStyle;
};