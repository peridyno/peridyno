#pragma once


#include <QtWidgets/QWidget>

#include "PortType.h"
#include "BlockData.h"
#include "Serializable.h"
#include "BlockGeometry.h"
#include "BlockStyle.h"
#include "BlockPainterDelegate.h"
#include "Export.h"
#include "memory.h"

namespace QtNodes
{

enum class ValidationState
{
  Valid,
  Warning,
  Error
};

class QtConnection;

class StyleCollection;

class NODE_EDITOR_PUBLIC QtBlockDataModel
  : public QObject
  , public Serializable
{
  Q_OBJECT

public:

  QtBlockDataModel();

  virtual
  ~QtBlockDataModel() = default;

  /// Caption is used in GUI
  virtual QString
  caption() const = 0;

  /// It is possible to hide caption in GUI
  virtual bool
  captionVisible() const { return true; }

  /// Port caption is used in GUI to label individual ports
  virtual QString
  portCaption(PortType, PortIndex) const { return QString(); }

  /// It is possible to hide port caption in GUI
  virtual bool
  portCaptionVisible(PortType, PortIndex) const { return false; }

  /// Name makes this model unique
  virtual QString
  name() const = 0;

public:

  QJsonObject
  save() const override;

public:

  virtual
  unsigned int nPorts(PortType portType) const = 0;

  virtual
  BlockDataType dataType(PortType portType, PortIndex portIndex) const = 0;

	std::shared_ptr<BlockData> portData(PortType portType, PortIndex portIndex);

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

  BlockStyle const&
  nodeStyle() const;

  void
  setNodeStyle(BlockStyle const& style);

public:

  /// Triggers the algorithm
  virtual
  void
  setInData(std::shared_ptr<BlockData> nodeData,
            PortIndex port) = 0;

  virtual
  std::shared_ptr<BlockData>
  outData(PortIndex port) = 0;

  virtual std::shared_ptr<BlockData> inData(PortIndex port) = 0;

  virtual
  QWidget *
  embeddedWidget() = 0;

  virtual
  bool
  resizable() const { return false; }

  virtual
  ValidationState
  validationState() const { return ValidationState::Valid; }

  virtual
  QString
  validationMessage() const { return QString(""); }

  virtual
  BlockPainterDelegate* painterDelegate() const { return nullptr; }

public Q_SLOTS:

  virtual void
  inputConnectionCreated(QtConnection const&)
  {
  }

  virtual void
  inputConnectionDeleted(QtConnection const&)
  {
  }

  virtual void
  outputConnectionCreated(QtConnection const&)
  {
  }

  virtual void
  outputConnectionDeleted(QtConnection const&)
  {
  }

Q_SIGNALS:

  void
  dataUpdated(PortIndex index);

  void
  dataInvalidated(PortIndex index);

  void
  computingStarted();

  void
  computingFinished();

  void embeddedWidgetSizeUpdated();

private:

  BlockStyle _nodeStyle;
};
}
