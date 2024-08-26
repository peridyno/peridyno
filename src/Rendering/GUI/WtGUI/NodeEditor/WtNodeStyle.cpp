#include "WtNodeStyle.h"

using json = nlohmann::json;

#define STYLE_READ_COLOR(values, variable)  { \
    auto valueRef = values[#variable]; \
	if(!valueRef.is_boolean()) { \
    if (valueRef.is_array()) { \
      std::vector<int> rgb; rgb.reserve(3); \
      for (auto it = valueRef.begin(); it != valueRef.end(); ++it) { \
        rgb.push_back(it->get<int>()); \
      } \
      variable.setRgb(rgb[0], rgb[1], rgb[2]); \
    } else { \
		std::string str = valueRef;\
		if(str.compare("white") == 0) {variable = Wt::WColor(Wt::StandardColor::White) ;}\
		if(str.compare("black") == 0) {variable = Wt::WColor(Wt::StandardColor::Black) ;}\
		if(str.compare("red") == 0) {variable = Wt::WColor(Wt::StandardColor::Red) ;}\
		if(str.compare("darkred") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkRed) ;}\
		if(str.compare("green") == 0) {variable = Wt::WColor(Wt::StandardColor::Green) ;}\
		if(str.compare("darkgreen") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkGreen) ;}\
		if(str.compare("blue") == 0) {variable = Wt::WColor(Wt::StandardColor::Blue) ;}\
		if(str.compare("darlblue") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkBlue) ;}\
		if(str.compare("cyan") == 0) {variable = Wt::WColor(Wt::StandardColor::Cyan) ;}\
		if(str.compare("darkcyan") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkCyan) ;}\
		if(str.compare("megenta") == 0) {variable = Wt::WColor(Wt::StandardColor::Magenta) ;}\
		if(str.compare("darkmegenta") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkMagenta) ;}\
		if(str.compare("yellow") == 0) {variable = Wt::WColor(Wt::StandardColor::Yellow) ;}\
		if(str.compare("darkyellow") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkYellow) ;}\
		if(str.compare("gray") == 0) {variable = Wt::WColor(Wt::StandardColor::Gray) ;}\
		if(str.compare("darkgray") == 0) {variable = Wt::WColor(Wt::StandardColor::DarkGray) ;}\
		if(str.compare("lightgray") == 0) {variable = Wt::WColor(Wt::StandardColor::LightGray) ;}\
		if(str.compare("transparent") == 0) {variable = Wt::WColor(Wt::StandardColor::Transparent) ;}\
		if(str.compare("lightcyan") == 0) {variable = Wt::WColor(224, 255, 255) ;}\
	} \
} \
}

#define STYLE_READ_FLOAT(values, variable)  { \
    auto valueRef = values[#variable]; \
	if(!valueRef.is_boolean()) { \
    if(valueRef.is_number_float()) { \
		variable = valueRef; \
	}\
	else { \
		Wt::log("error") << "Error with Json"; \
	}\
} \
}

#define STYLE_READ_BOOL(values, variable)  { \
    auto valueRef = values[#variable]; \
	if(valueRef.is_boolean()){ \
		variable = valueRef; \
	}\
	else { \
		Wt::log("error") << "Error with Json"; \
	}\
}

WtNodeStyle::WtNodeStyle()
{
	std::string filePath = getAssetPath() + "../external/nodeeditor/resources/DefaultStyle.json";
	loadJsonFile(filePath);
}

WtNodeStyle::WtNodeStyle(std::string jsonText)
{
	loadJsonText(jsonText);
}

void WtNodeStyle::setNodeStyle(std::string jsonText)
{
	WtNodeStyle style(jsonText);
	//WtStyleCollection::setNodeStyle(style);
}

void WtNodeStyle::loadJsonFile(std::string styleFile)
{
	std::ifstream file(styleFile, std::ios::in);

	if (!file.is_open())
	{
		Wt::log("error") << "Couldn't open style file" << styleFile;
		return;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	loadJsonFromByteArray(buffer.str());
	file.close();
}

void WtNodeStyle::loadJsonText(std::string jsonText)
{
	loadJsonFromByteArray(jsonText);
}

void WtNodeStyle::loadJsonFromByteArray(std::string const& jsonData)
{
	json j = json::parse(jsonData);
	json obj = j.at("NodeStyle");
	STYLE_READ_COLOR(obj, NormalBoundaryColor);
	STYLE_READ_COLOR(obj, SelectedBoundaryColor);
	STYLE_READ_COLOR(obj, GradientColor0);
	STYLE_READ_COLOR(obj, GradientColor1);
	STYLE_READ_COLOR(obj, GradientColor2);
	STYLE_READ_COLOR(obj, GradientColor3);
	STYLE_READ_COLOR(obj, ShadowColor);
	STYLE_READ_COLOR(obj, FontColor);
	STYLE_READ_COLOR(obj, FontColorFaded);
	STYLE_READ_COLOR(obj, ConnectionPointColor);
	STYLE_READ_COLOR(obj, FilledConnectionPointColor);
	STYLE_READ_COLOR(obj, WarningColor);
	STYLE_READ_COLOR(obj, ErrorColor);

	STYLE_READ_COLOR(obj, HotKeyColor0);
	STYLE_READ_COLOR(obj, HotKeyColor1);
	STYLE_READ_COLOR(obj, HotKeyColor2);
	STYLE_READ_COLOR(obj, HotKeyColor1);

	STYLE_READ_FLOAT(obj, PenWidth);
	STYLE_READ_FLOAT(obj, HoveredPenWidth);
	STYLE_READ_FLOAT(obj, ConnectionPointDiameter);

	STYLE_READ_FLOAT(obj, Opacity);
}

WtConnectionStyle::WtConnectionStyle()
{
	std::string filePath = getAssetPath() + "../external/nodeeditor/resources/DefaultStyle.json";
	loadJsonFile(filePath);
}

WtConnectionStyle::WtConnectionStyle(std::string jsonText)
{
	loadJsonText(jsonText);
}

void WtConnectionStyle::setConnectionStyle(std::string jsonText)
{
	WtConnectionStyle style(jsonText);
	//WtStyleCollection::setConnectionStyle(style);
}

void WtConnectionStyle::loadJsonFile(std::string styleFile)
{
	std::ifstream file(styleFile, std::ios::in);

	if (!file.is_open())
	{
		Wt::log("error") << "Couldn't open style file" << styleFile;
		return;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	loadJsonFromByteArray(buffer.str());
	file.close();
}

void WtConnectionStyle::loadJsonText(std::string jsonText)
{
	loadJsonFromByteArray(jsonText);
}

void WtConnectionStyle::loadJsonFromByteArray(std::string const& jsonData)
{
	json j = json::parse(jsonData);
	json obj = j.at("ConnectionStyle");
	STYLE_READ_COLOR(obj, ConstructionColor);
	STYLE_READ_COLOR(obj, NormalColor);
	STYLE_READ_COLOR(obj, SelectedColor);
	STYLE_READ_COLOR(obj, SelectedHaloColor);
	STYLE_READ_COLOR(obj, HoveredColor);

	STYLE_READ_FLOAT(obj, LineWidth);
	STYLE_READ_FLOAT(obj, ConstructionLineWidth);
	STYLE_READ_FLOAT(obj, PointDiameter);

	STYLE_READ_BOOL(obj, UseDataDefinedColors);
}

Wt::WColor WtConnectionStyle::constructionColor() const
{
	return ConstructionColor;
}

Wt::WColor WtConnectionStyle::normalColor() const
{
	return NormalColor;
}

Wt::WColor WtConnectionStyle::normalColor(std::string typeID) const
{

	std::hash<std::string> hasher;

	std::size_t hash_value = hasher(typeID);

	std::mt19937 gen(static_cast<unsigned int>(hash_value));

	std::uniform_int_distribution<> hueDist(0, 255);

	int hue = hueDist(gen);

	int sat = 120 + hash_value % 129;

	return fromHSL(hash_value, sat, 160);
}

Wt::WColor WtConnectionStyle::fromHSL(int h, int s, int l) const
{
	double hue = h / 255.0;
	double saturation = s / 255.0;
	double lightness = l / 255.0;

	double c = (1 - abs(2 * lightness - 1)) * saturation;
	double x = c * (1 - abs(fmod(hue * 6, 2) - 1));
	double m = lightness - c / 2;

	double r = 0, g = 0, b = 0;

	if (0 <= hue && hue < 1 / 6) {
		r = c; g = x; b = 0;
	}
	else if (1 / 6 <= hue && hue < 1 / 3) {
		r = x; g = c; b = 0;
	}
	else if (1 / 3 <= hue && hue < 1 / 2) {
		r = 0; g = c; b = x;
	}
	else if (1 / 2 <= hue && hue < 2 / 3) {
		r = 0; g = x; b = c;
	}
	else if (2 / 3 <= hue && hue < 5 / 6) {
		r = x; g = 0; b = c;
	}
	else if (5 / 6 <= hue && hue < 1) {
		r = c; g = 0; b = x;
	}

	r = (r + m) * 255;
	g = (g + m) * 255;
	b = (b + m) * 255;

	return Wt::WColor(r, g, b);
}

Wt::WColor WtConnectionStyle::selectedColor() const
{
	return SelectedColor;
}

Wt::WColor WtConnectionStyle::selectedHaloColor() const
{
	return SelectedHaloColor;
}

Wt::WColor WtConnectionStyle::hoveredColor() const
{
	return HoveredColor;
}

float WtConnectionStyle::lineWidth() const
{
	return LineWidth;
}

float WtConnectionStyle::constructionLineWidth() const
{
	return ConstructionLineWidth;
}

float WtConnectionStyle::pointDiameter() const
{
	return PointDiameter;
}

bool WtConnectionStyle::useDataDefinedColors() const
{
	return UseDataDefinedColors;
}

WtFlowViewStyle::WtFlowViewStyle()
{
	std::string filePath = getAssetPath() + "../external/nodeeditor/resources/DefaultStyle.json";
	loadJsonFile(filePath);
}

WtFlowViewStyle::WtFlowViewStyle(std::string jsonText)
{
	loadJsonText(jsonText);
}

void WtFlowViewStyle::setStyle(std::string jsonText)
{
	WtFlowViewStyle style(jsonText);
	//WtStyleCollection::setNodeStyle(style);
}

void WtFlowViewStyle::loadJsonFile(std::string styleFile)
{
	std::ifstream file(styleFile, std::ios::in);

	if (!file.is_open())
	{
		Wt::log("error") << "Couldn't open style file" << styleFile;
		return;
	}
	std::stringstream buffer;
	buffer << file.rdbuf();
	loadJsonFromByteArray(buffer.str());
	file.close();
}

void WtFlowViewStyle::loadJsonText(std::string jsonText)
{
	loadJsonFromByteArray(jsonText);
}

void WtFlowViewStyle::loadJsonFromByteArray(std::string const& jsonData)
{
	json j = json::parse(jsonData);
	json obj = j.at("FlowViewStyle");
	STYLE_READ_COLOR(obj, BackgroundColor);
	STYLE_READ_COLOR(obj, FineGridColor);
	STYLE_READ_COLOR(obj, CoarseGridColor);
}

WtNodeStyle const& WtStyleCollection::nodeStyle()
{
	return instance()._nodeStyle;
}

WtConnectionStyle const& WtStyleCollection::connectionStyle()
{
	return instance()._connectionStyle;
}

WtFlowViewStyle const& WtStyleCollection::flowViewStyle()
{
	return instance()._flowViewStyle;
}

void WtStyleCollection::setNodeStyle(WtNodeStyle nodeStyle)
{
	instance()._nodeStyle = nodeStyle;
}

void WtStyleCollection::setConnectionStyle(WtConnectionStyle connectionStyle)
{
	instance()._connectionStyle = connectionStyle;
}

void WtStyleCollection::setFlowViewStyle(WtFlowViewStyle flowViewStyle)
{
	instance()._flowViewStyle = flowViewStyle;
}

WtStyleCollection& WtStyleCollection::instance()
{
	static WtStyleCollection collection;

	return collection;
}