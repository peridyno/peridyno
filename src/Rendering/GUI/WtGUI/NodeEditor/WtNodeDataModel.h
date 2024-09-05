#pragma once

#include "WtNodeData.hpp"
#include "WtNodeStyle.h"

enum class NodeValidationState
{
	Valid,
	Warning,
	Error
};

class WtConnection;
class WtNodePainterDelegate;

class WtNodeDataModel
{
public:
	WtNodeDataModel();

	virtual ~WtNodeDataModel() = default;

	virtual std::string caption() const = 0;

	virtual std::string nodeTips() const { return "nodeTips"; }

	virtual std::string portTips(PortType, PortIndex) const { return "portTips"; }

	/// It is possible to hide caption in GUI
	virtual bool captionVisible() const { return true; }

	virtual bool hotkeyEnabled() const { return true; }

	/// Port caption is used in GUI to label individual ports
	virtual std::string portCaption(PortType, PortIndex) const { return std::string(); }

	/// It is possible to hide port caption in GUI
	virtual bool portCaptionVisible(PortType, PortIndex) const { return false; }

	virtual bool allowExported() const { return true; }

	/// Name makes this model unique
	virtual std::string name() const = 0;

public:
	//QJsonObject save() const override;

public:
	virtual unsigned int nPorts(PortType portType) const = 0;

	virtual NodeDataType dataType(PortType portType, PortIndex portIndex) const = 0;

public:
	enum class ConnectionPolicy
	{
		One,
		Many,
	};

	virtual ConnectionPolicy portOutConnectionPolicy(PortIndex) const
	{
		return ConnectionPolicy::Many;
	}

	virtual ConnectionPolicy portInConnectionPolicy(PortIndex) const
	{
		return ConnectionPolicy::One;
	}

	WtNodeStyle const& nodeStyle() const;

	void setNodeStyle(WtNodeStyle const& style);

public:
	/// Triggers the algorithm
	virtual	void setInData(std::shared_ptr<WtNodeData> nodeData, PortIndex port) = 0;

	virtual bool tryInData(PortIndex portIndex, std::shared_ptr<WtNodeData> nodeData) { return true; }

	virtual std::shared_ptr<WtNodeData> outData(PortIndex port) = 0;

	//virtual QWidget* embeddedWidget() = 0;

	virtual bool resizable() const { return false; }

	virtual NodeValidationState validationState() const { return NodeValidationState::Valid; }

	virtual std::string validationMessage() const { return std::string(""); }

	virtual WtNodePainterDelegate* painterDelegate() const { return nullptr; }

public:
	virtual void inputConnectionCreated(WtConnection const&) {}

	virtual void inputConnectionDeleted(WtConnection const&) {}

	virtual void outputConnectionCreated(WtConnection const&) {}

	virtual void outputConnectionDeleted(WtConnection const&) {}

	//signal
	//void dataUpdated(PortIndex index);

	//void dataInvalidated(PortIndex index);

	//void computingStarted();

	//void computingFinished();

	//void embeddedWidgetSizeUpdated();

private:
	WtNodeStyle _nodeStyle;
};