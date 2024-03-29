#include "QPiecewiseSpinBox.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QValueDialog.h"

namespace dyno
{
	QPiecewiseSpinBox::QPiecewiseSpinBox(QWidget* parent)
		: QSpinBox(parent)
	{
		this->setKeyboardTracking(false);
		mValueModify = new QValueDialog(this);
	}

	void QPiecewiseSpinBox::wheelEvent(QWheelEvent* event)
	{

	}

	void QPiecewiseSpinBox::contextMenuEvent(QContextMenuEvent* event)
	{
		if(mValueModify== nullptr)
			mValueModify = new QValueDialog(this);
		mValueModify->updateDialogPosition();
		mValueModify->show();
	}



}

