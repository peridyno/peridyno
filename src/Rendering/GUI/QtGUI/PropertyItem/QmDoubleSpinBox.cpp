#include "QmDoubleSpinBox.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QmValueDialog.h"

namespace dyno
{


	mDoubleSpinBox::mDoubleSpinBox(QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		//this->lineEdit()->setMouseTracking(true);

		connect(this->lineEdit(), SIGNAL(textChanged(const QString&)), this, SLOT(LineEditStart()));
		connect(this, SIGNAL(valueChanged(double)), this, SLOT(LineEditFinished(double)));

		this->setDecimals(decimalsMax);
		this->setKeyboardTracking(false);

	}

	void mDoubleSpinBox::LineEditStart()
	{

		return;
	}

	void mDoubleSpinBox::LineEditFinished(double v) 
	{

		realValue = v;
		this->setValue(realValue);

		return;
	}

	void mDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{

	}
	void mDoubleSpinBox::contextMenuEvent(QContextMenuEvent* event) 
	{

		if(ValueModify== nullptr)
			ValueModify = new ValueDialog(this);
		ValueModify->updateDialogPosition();
		ValueModify->show();
	}


	void mDoubleSpinBox::mousePressEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mousePressEvent(event);

	}

	void mDoubleSpinBox::mouseReleaseEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseReleaseEvent(event);
	}

	void mDoubleSpinBox::mouseMoveEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseMoveEvent(event);

	}

	void mDoubleSpinBox::ModifyValue(double v)
	{
		this->setRealValue(v);

	}

	void mDoubleSpinBox::ModifyValueAndUpdate(double v)
	{
		this->setKeyboardTracking(true);
		this->setRealValue(v);
		this->lineEdit()->setText(QString::number(realValue, 10, displayDecimals));
		this->setKeyboardTracking(false);
	}


}

