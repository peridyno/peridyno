#include "StyleCollection.h"

using QtNodes::StyleCollection;
using QtNodes::BlockStyle;
using QtNodes::ConnectionStyle;
using QtNodes::FlowViewStyle;

BlockStyle const&
StyleCollection::
nodeStyle()
{
  return instance()._nodeStyle;
}


ConnectionStyle const&
StyleCollection::
connectionStyle()
{
  return instance()._connectionStyle;
}


FlowViewStyle const&
StyleCollection::
flowViewStyle()
{
  return instance()._flowViewStyle;
}


void
StyleCollection::
setNodeStyle(BlockStyle nodeStyle)
{
  instance()._nodeStyle = nodeStyle;
}


void
StyleCollection::
setConnectionStyle(ConnectionStyle connectionStyle)
{
  instance()._connectionStyle = connectionStyle;
}


void
StyleCollection::
setFlowViewStyle(FlowViewStyle flowViewStyle)
{
  instance()._flowViewStyle = flowViewStyle;
}



StyleCollection&
StyleCollection::
instance()
{
  static StyleCollection collection;

  return collection;
}
