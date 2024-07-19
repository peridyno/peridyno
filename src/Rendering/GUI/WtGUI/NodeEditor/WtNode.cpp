#include "WtNode.h"

#include <utility>
#include <iostream>
#include <cmath>
#include <Wt/WPainter.h>
#include <Wt/WFont.h>
#include <Wt/WFontMetrics.h>
#include <Wt/WLabel.h>

#include "WtNodeGraphicsObject.h"
#include "WtNodeDataModel.h"


NodeGeometry::NodeGeometry(std::unique_ptr<WtNodeDataModel> const& dataModel)
	: _width(100)
	, _height(150)
	, _inputPortWidth(70)
	, _outputPortWidth(70)
	, _entryHeight(20)
	, _spacing(20)
	, _hotkeyWidth(20)
	, _hotkeyIncline(5)
	, _hotkeyOffset(15)
	, _captionHeightSpacing(5)
	, _hovered(false)
	, _nSources(dataModel->nPorts(PortType::Out))
	, _nSinks(dataModel->nPorts(PortType::In))
	, _draggingPos(-1000, -1000)
	, _dataModel(dataModel)
	//, _fontMetrics(Wt::WFont())
	//, _boldFontMetrics(Wt::WFont())
	//, TipsWidget(new QDialog)
	//, TipsWidget_(new QDockWidget)
	//, PortTipsWidget(new QDialog)
	, isPortTipsShow(new bool(false))
	, isNodeTipsShow(new bool(false))
{
	Wt::WFont f;
	////f.setBold(true);
	////this->InitText();
	//_boldFontMetrics = Wt::WFontMetrics(f);

}

NodeGeometry::~NodeGeometry()
{
	//HidePortTips();
}

void NodeGeometry::ShowTips()const
{
	//*isNodeTipsShow = true;

	//Wt::WLabel* TipsLabel = new Wt::WLabel;
	//TipsLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	//TipsLabel->setParent(TipsWidget);
	//TipsLabel->setText(_dataModel->nodeTips());
	//TipsLabel->setWordWrap(true);
	//TipsLabel->setFixedWidth(200);
	//TipsLabel->adjustSize();
	//TipsLabel->setStyleSheet("QLabel{color:white;background-color:#346792;border: 1px solid #000000;border-radius:3px; padding: 0px;}");

	//TipsWidget->setWindowFlags(Qt::WindowTransparentForInput | Qt::ToolTip);
	//TipsWidget->move(QCursor().pos().x() + 20, QCursor().pos().y() + 20);
	//if (!_dataModel->nodeTips().isEmpty())
	//{
	//	TipsWidget->show();
	//}
}

void NodeGeometry::HideTips()const
{
	//*isNodeTipsShow = false;
	//TipsWidget->hide();
	//delete TipsWidget->findChild<QLabel*>();
}

void NodeGeometry::HidePortTips()const
{
	//if (*isPortTipsShow)
	//	*isPortTipsShow = false;
	//PortTipsWidget->hide();
	//delete PortTipsWidget->findChild<QLabel*>();
}

bool NodeGeometry::getPortTipsState()const { return *isPortTipsShow; }

void NodeGeometry::ShowPortTips()const
{
	//if (!*isPortTipsShow && !*isNodeTipsShow)
	//{
	//	*isPortTipsShow = true;

	//	QLabel* PortTipsLabel = new QLabel;
	//	PortTipsLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	//	PortTipsLabel->setParent(PortTipsWidget);
	//	PortTipsLabel->setText(_dataModel->portTips(hoverport_type, hoverport_id));
	//	PortTipsLabel->setWordWrap(true);

	//	//PortTipsLabel->setFixedWidth(200);
	//	PortTipsLabel->adjustSize();
	//	PortTipsLabel->setStyleSheet("QLabel{color:white;background-color:#346792;border: 1px solid #000000;border-radius:3px ; padding: 0px;}");
	//	PortTipsWidget->setWindowFlags(Qt::WindowTransparentForInput | Qt::ToolTip);

	//	PortTipsWidget->move(QCursor().pos().x() + 20, QCursor().pos().y() + 20);

	//	if (!_dataModel->portTips(hoverport_type, hoverport_id).isEmpty())
	//	{
	//		PortTipsWidget->show();
	//	}

	//}

}

unsigned int NodeGeometry::nSources() const
{
	return _dataModel->nPorts(PortType::Out);
}

unsigned int NodeGeometry::nSinks() const
{
	return _dataModel->nPorts(PortType::In);
}

bool NodeGeometry::checkHitHotKey0(Wt::WPointF point, Wt::WTransform const& t /*= Wt::WTransform()*/) const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	float diam = nodeStyle.ConnectionPointDiameter;

	Wt::WPointF p0 = t.map(Wt::WPointF(width() + diam, -diam));

	float x = p0.x() - point.x();
	float y = point.y() - p0.y();

	float h = diam + captionHeight();

	if (y > 0.0f && y < h)
	{
		float ext = hotkeyIncline() * y / h;

		if (x > hotkeyOffset() && x < hotkeyWidth() + hotkeyOffset() + ext)
		{
			return true;
		}
	}
	return false;
}

bool NodeGeometry::checkHitHotKey1(Wt::WPointF point, Wt::WTransform const& t /*= Wt::WTransform()*/) const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	float diam = nodeStyle.ConnectionPointDiameter;

	Wt::WPointF p0 = t.map(Wt::WPointF(width() + diam, -diam));

	float x = p0.x() - point.x();
	float y = point.y() - p0.y();

	float h = diam + captionHeight();

	if (y > 0.0f && y < h)
	{
		float ext = hotkeyIncline() * y / h;

		if (x > hotkeyWidth() + hotkeyOffset() && x < 2 * hotkeyWidth() + hotkeyOffset() + ext)
		{
			return true;
		}
	}
	return false;
}

//TODO:need transplant

Wt::WRectF NodeGeometry::entryBoundingRect() const
{
	double const addon = 0.0;

	return Wt::WRectF(0 - addon,
		0 - addon,
		_entryWidth + 2 * addon,
		_entryHeight + 2 * addon);
}

Wt::WRectF NodeGeometry::boundingRect() const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	double addon = 4 * nodeStyle.ConnectionPointDiameter;

	return Wt::WRectF(0 - addon,
		0 - addon,
		_width + 2 * addon,
		_height + 2 * addon);
}

void NodeGeometry::recalculateSize() const
{
	//_entryHeight = _fontMetrics.height();

	//{
	//	unsigned int maxNumOfEntries = std::max(_nSinks, _nSources);
	//	unsigned int step = _entryHeight + _spacing;
	//	_height = step * maxNumOfEntries;
	//}

	//if (auto w = _dataModel->embeddedWidget())
	//{
	//	_height = std::max(_height, static_cast<unsigned>(w->height()));
	//}

	//_height += captionHeight();

	//_inputPortWidth = portWidth(PortType::In);
	//_outputPortWidth = portWidth(PortType::Out);

	//_width = _inputPortWidth +
	//	_outputPortWidth +
	//	2 * _spacing;

	//if (auto w = _dataModel->embeddedWidget())
	//{
	//	_width += w->width();
	//}

	//_width = std::max(_width, captionWidth());

	//if (_dataModel->validationState() != NodeValidationState::Valid)
	//{
	//	_width = std::max(_width, validationWidth());
	//	_height += validationHeight() + _spacing;
	//}
}


void
NodeGeometry::
recalculateSize(Wt::WFont const& font) const
{
	//Wt::WFontMetrics fontMetrics(font);
	//Wt::WFont boldFont = font;

	//boldFont.setBold(true);

	//Wt::WFontMetrics boldFontMetrics(boldFont);

	//if (_boldFontMetrics != boldFontMetrics)
	//{
	//	_fontMetrics = fontMetrics;
	//	_boldFontMetrics = boldFontMetrics;

	//	recalculateSize();
	//}
}

Wt::WPointF NodeGeometry::portScenePosition(PortType index, PortType portType, Wt::WTransform const& t) const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	unsigned int step = _entryHeight + _spacing;

	Wt::WPointF result;

	double totalHeight = captionHeightSpacing();

	totalHeight += captionHeight();

	//totalHeight += step * index;

	// TODO: why?
	totalHeight += step / 2.0;

	switch (portType)
	{
	case PortType::Out:
	{
		double x = _width + nodeStyle.ConnectionPointDiameter;

		result = _dataModel->allowExported() ? Wt::WPointF(x, totalHeight - captionHeight() - captionHeightSpacing() - step / 3.0) : Wt::WPointF(x, totalHeight);
		break;
	}

	case PortType::In:
	{
		double x = 0.0 - nodeStyle.ConnectionPointDiameter;

		result = Wt::WPointF(x, totalHeight);
		break;
	}

	default:
		break;
	}

	return t.map(result);
}


PortIndex NodeGeometry::checkHitScenePoint(PortType portType, Wt::WPointF const scenePoint, Wt::WTransform const& sceneTransform) const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	PortIndex result = INVALID_PORT;

	if (portType == PortType::None)
		return result;

	double const tolerance = 2.0 * nodeStyle.ConnectionPointDiameter;

	unsigned int const nItems = _dataModel->nPorts(portType);

	//for (unsigned int i = 0; i < nItems; ++i)
	//{
	//	auto pp = portScenePosition(i, portType, sceneTransform);
	//	Wt::WPointF p = pp - scenePoint;
	//	auto    distance = std::sqrt(Wt::WPointF::dotProduct(p, p));
	//	if (distance < tolerance)
	//	{
	//		result = PortIndex(i);
	//		break;
	//	}
	//}

	return result;
}


PortIndex NodeGeometry::hoverHitScenePoint(PortType portType, Wt::WPointF const scenePoint, Wt::WTransform const& sceneTransform) const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	PortIndex result = INVALID_PORT;

	if (portType == PortType::None)
		return result;

	double const tolerance = 2.0 * nodeStyle.ConnectionPointDiameter;

	unsigned int const nItems = _dataModel->nPorts(portType);

	//for (unsigned int i = 0; i < nItems; ++i)
	//{
	//	auto pp = portScenePosition(i, portType, sceneTransform);
	//	Wt::WPointF p = pp - scenePoint;
	//	auto    distance = std::sqrt(Wt::WPointF::dotProduct(p, p));
	//	if (distance < tolerance)
	//	{
	//		result = PortIndex(i);
	//		break;
	//	}
	//}

	return result;
}



PortIndex NodeGeometry::hoverHitPortArea(
	PortType portType,
	Wt::WPointF const scenePoint,
	Wt::WTransform const& sceneTransform,
	NodeGeometry const& geom,
	WtNodeDataModel const* model) const
{
	auto const& nodeStyle = WtStyleCollection::nodeStyle();

	//Wt::WFont font;
	////font.setFamily("none");
	//Wt::WFontMetrics metrics(font);

	PortIndex result = INVALID_PORT;

	if (portType == PortType::None)
		return result;

	//Wt::WPointF p1;
	//Wt::WPointF p2;

	//unsigned int const nItems = _dataModel->nPorts(portType);

	////std::string type;
	//for (size_t i = 0; i < nItems; ++i)
	//{
	//	p1 = geom.portScenePosition(i, portType, sceneTransform);
	//	Wt::WPointF pt = Wt::WPointF(geom.portScenePosition(i, portType, sceneTransform));
	//	//if (portType == PortType::None)
	//	//{
	//	//	type = "None";
	//	//}
	//	//else if (portType == PortType::In)
	//	//{
	//	//	type = "In";
	//	//}
	//	//else if (portType == PortType::Out)
	//	//{
	//	//	type = "Out";
	//	//}
	//	std::string s;

	//	if (model->portCaptionVisible(portType, (PortIndex)i))
	//	{
	//		s = model->portCaption(portType, (PortIndex)i);
	//	}
	//	else
	//	{
	//		s = model->dataType(portType, (PortIndex)i).name;
	//	}

	//	auto rect = metrics.boundingRect(s);

	//	p2.setY(p1.y() + rect.height() / 4);
	//	p1.setY(p1.y() - rect.height() * 3 / 5);

	//	switch (portType)
	//	{
	//	case PortType::In:
	//		p1.setX(0 + sceneTransform.dx());
	//		p2.setX(rect.width() + sceneTransform.dx());
	//		break;

	//	case PortType::Out:
	//		p1.setX(geom.width() - rect.width() + sceneTransform.dx());
	//		p2.setX(geom.width() + sceneTransform.dx());
	//		break;

	//	default:
	//		break;
	//	}
	//	//std::cout << type <<":" << i << "--" << "p1\B5\C4λ\D6\C3   x=" << p1.x() << "   y=" << p1.y() << std::endl;
	//	//std::cout << type <<":" << i << "--" << "p2\B5\C4λ\D6\C3   x=" << p2.x() << "   y=" << p2.y() << std::endl;

	//	if (scenePoint.x() > p1.x() && scenePoint.x() < p2.x() && scenePoint.y() > p1.y() && scenePoint.y() < p2.y())
	//	{
	//		//std::cout <<"\B7\B5\BB\D8ֵ\A3\BA" << type << i << std::endl;
	//		//std::cout << "********************* <  \D2ѷ\B5\BB\D8  > *********************"  << std::endl;
	//		return PortIndex(i);
	//		//std::cout << "********************* <  break  > *********************" << std::endl;
	//		break;
	//	}

	//}
	return result = INVALID_PORT;
}

PortIndex NodeGeometry::findHitPort(PortType portType,
	Wt::WPointF const scenePoint,
	Wt::WTransform const& sceneTransform,
	NodeGeometry const& geom,
	WtNodeDataModel const* model) const
{
	PortIndex result = INVALID_PORT;
	PortType _portType = portType;
	Wt::WPointF const _scenePoint = scenePoint;
	Wt::WTransform const _sceneTransform = sceneTransform;
	NodeGeometry const _geom = geom;
	WtNodeDataModel const* _model = model;

	result = geom.hoverHitPortArea(_portType, _scenePoint, _sceneTransform, _geom, _model);
	//std::cout << "result1=" << result << std::endl;
	if (result == INVALID_PORT)
	{
		result = geom.hoverHitScenePoint(_portType, _scenePoint, _sceneTransform);
		//std::cout << "result2=" << result << std::endl;
	}

	return result;

}

//QRect
//NodeGeometry::
//resizeRect() const
//{
//	unsigned int rectSize = 7;
//
//	return QRect(_width - rectSize,
//		_height - rectSize,
//		rectSize,
//		rectSize);
//}

Wt::WPointF NodeGeometry::widgetPosition() const
{
	//if (auto w = _dataModel->embeddedWidget())
	//{
	//	if (w->sizePolicy().verticalPolicy() & QSizePolicy::ExpandFlag)
	//	{
	//		// If the widget wants to use as much vertical space as possible, place it immediately after the caption.
	//		return Wt::WPointF(_spacing + portWidth(PortType::In), captionHeight());
	//	}
	//	else
	//	{
	//		if (_dataModel->validationState() != NodeValidationState::Valid)
	//		{
	//			return Wt::WPointF(_spacing + portWidth(PortType::In),
	//				(captionHeight() + _height - validationHeight() - _spacing - w->height()) / 2.0);
	//		}

	//		return Wt::WPointF(_spacing + portWidth(PortType::In),
	//			(captionHeight() + _height - w->height()) / 2.0);
	//	}
	//}
	return Wt::WPointF();
}

int NodeGeometry::equivalentWidgetHeight() const
{
	if (_dataModel->validationState() != NodeValidationState::Valid)
	{
		return height() - captionHeight() + validationHeight();
	}

	return height() - captionHeight();
}

unsigned int NodeGeometry::captionHeight() const
{
	//if (!_dataModel->captionVisible())
	//	return 0;

	//std::string name = _dataModel->caption();

	//return _boldFontMetrics.boundingRect(name).height() + captionHeightSpacing();
	return captionHeightSpacing();
}


unsigned int
NodeGeometry::
captionWidth() const
{
	if (!_dataModel->captionVisible())
		return 0;

	//std::string name = _dataModel->caption();

	//unsigned int w = _dataModel->hotkeyEnabled() ? 2 * captionHeight() + _boldFontMetrics.boundingRect(name).width() + 2 * hotkeyWidth() + hotkeyIncline()
	//	: 2 * captionHeight() + _boldFontMetrics.boundingRect(name).width();

	//return w;
}


unsigned int
NodeGeometry::
validationHeight() const
{
	//std::string msg = _dataModel->validationMessage();

	//return _boldFontMetrics.boundingRect(msg).height();
	return 0;
}

unsigned int
NodeGeometry::
validationWidth() const
{
	//std::string msg = _dataModel->validationMessage();

	//return _boldFontMetrics.boundingRect(msg).width();
	return 0;
}


//Wt::WPointF
//NodeGeometry::
//calculateNodePositionBetweenNodePorts(PortIndex targetPortIndex, PortType targetPort, WtNode* targetNode,
//	PortIndex sourcePortIndex, PortType sourcePort, WtNode* sourceNode,
//	WtNode& newNode)
//{
//	//Calculating the nodes position in the scene. It'll be positioned half way between the two ports that it "connects". 
//	//The first line calculates the halfway point between the ports (node position + port position on the node for both nodes averaged).
//	//The second line offsets this coordinate with the size of the new node, so that the new nodes center falls on the originally
//	//calculated coordinate, instead of it's upper left corner.
//	// 
//	//auto converterNodePos = (sourceNode->nodeGraphicsObject().pos() + sourceNode->nodeGeometry().portScenePosition(sourcePortIndex, sourcePort) +
//	//	targetNode->nodeGraphicsObject().pos() + targetNode->nodeGeometry().portScenePosition(targetPortIndex, targetPort)) / 2.0f;
//	//converterNodePos.setX(converterNodePos.x() - newNode.nodeGeometry().width() / 2.0f);
//	//converterNodePos.setY(converterNodePos.y() - newNode.nodeGeometry().height() / 2.0f);
//	//return converterNodePos;
//}


unsigned int
NodeGeometry::
portWidth(PortType portType) const
{
	unsigned width = 0;

	//for (auto i = 0ul; i < _dataModel->nPorts(portType); ++i)
	//{
	//	std::string name;

	//	if (_dataModel->portCaptionVisible(portType, i))
	//	{
	//		name = _dataModel->portCaption(portType, i);
	//	}
	//	else
	//	{
	//		name = _dataModel->dataType(portType, i).name;
	//	}
	//	width = std::max(unsigned(_fontMetrics.boundingRect(name).width()),
	//		width);
	//}

	return width;
}

NodeGeometry& WtNode::nodeGeometry()
{
	return _nodeGeometry;
}
