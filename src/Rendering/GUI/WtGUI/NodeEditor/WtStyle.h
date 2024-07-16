#pragma once

#include <Wt/WColor.h>

class Style
{
public:
	virtual ~Style() = default;

private:
	virtual void loadJsonText(std::string jsonText) = 0;
	virtual void loadJsonFile(std::string fileName) = 0;
	virtual void loadJsonFromByteArray(std::string const& byteArray) = 0;
};

class NodeStyle : public Style
{
public:
	NodeStyle();

	NodeStyle(std::string jsonText);

public:
	static void setNodeStyle(std::string jsonText);

private:
	void loadJsonText(std::string jsonText) override;
	void loadJsonFile(std::string fileName) override;
	void loadJsonFromByteArray(std::string const& byteArray) override;

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