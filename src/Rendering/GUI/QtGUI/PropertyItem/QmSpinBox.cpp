#include "QmSpinBox.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QmValueDialog.h"

namespace dyno
{
	mSpinBox::mSpinBox(QWidget* parent)
		: QSpinBox(parent)
	{
		this->setKeyboardTracking(false);
		ValueModify = new ValueDialog(this);
	}

	void mSpinBox::wheelEvent(QWheelEvent* event)
	{

	}

	void mSpinBox::contextMenuEvent(QContextMenuEvent* event)
	{
		if(ValueModify== nullptr)
			ValueModify = new ValueDialog(this);
		ValueModify->updateDialogPosition();
		ValueModify->show();
	}



}

