#pragma once

#include <QtCore/QString>

#include "Export.h"

namespace QtNodes
{

struct BlockDataType
{
  QString id;
  QString name;
};

/// Class represents data transferred between nodes.
/// @param type is used for comparing the types
/// The actual data is stored in subtypes
class NODE_EDITOR_PUBLIC BlockData
{
public:

  virtual ~BlockData() = default;

  virtual bool sameType(BlockData const &nodeData) const
  {
    return (this->type().id == nodeData.type().id);
  }

  virtual bool isKindOf(BlockData &nodedata) const = 0;

  /// Type for inner use
  virtual BlockDataType type() const = 0;

  bool isToDisconnected() {
	  return m_isToDisconnected;
  }

  void setDisconnected(bool connected)
  {
	  m_isToDisconnected = connected;
  }

private:
	bool m_isToDisconnected = false;
};
}
