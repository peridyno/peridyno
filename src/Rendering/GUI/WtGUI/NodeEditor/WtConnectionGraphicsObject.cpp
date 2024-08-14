#include "WtConnectionGraphicsObject.h"

#include "WtFlowScene.h"

WtConnectionGraphicsObject::WtConnectionGraphicsObject(WtFlowScene& scene, WtConnection& connection)
	: _scene(scene)
	, _connection(connection)
{
	//_scene.addItem(this);
}

WtConnectionGraphicsObject::~WtConnectionGraphicsObject() {}