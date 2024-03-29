#include "QPiecewiseDoubleSpinBox.h"

#include "Module.h"
#include "Node.h"
#include "Field.h"
#include "SceneGraphFactory.h"

#include <QVBoxLayout>

#include <memory>
#include "QValueDialog.h"

namespace dyno
{


	QPiecewiseDoubleSpinBox::QPiecewiseDoubleSpinBox(QWidget* parent)
		: QDoubleSpinBox(parent)
	{
		//this->lineEdit()->setMouseTracking(true);

		connect(this->lineEdit(), SIGNAL(textChanged(const QString&)), this, SLOT(LineEditStart()));
		connect(this, SIGNAL(valueChanged(double)), this, SLOT(LineEditFinished(double)));
		connect(this, SIGNAL(valueChanged(double)), this, SLOT(ModifyValueAndUpdate(double)));
		
		this->setDecimals(decimalsMax);
		this->setKeyboardTracking(false);

	}

	void QPiecewiseDoubleSpinBox::LineEditStart()
	{
		return;
	}

	void QPiecewiseDoubleSpinBox::LineEditFinished(double v) 
	{
		realValue = v;
		this->setValue(realValue);

		return;
	}

	void QPiecewiseDoubleSpinBox::wheelEvent(QWheelEvent* event)
	{

	}
	void QPiecewiseDoubleSpinBox::contextMenuEvent(QContextMenuEvent* event) 
	{

		if(ValueModify== nullptr)
			ValueModify = new QValueDialog(this);
		ValueModify->updateDialogPosition();
		ValueModify->show();
	}


	void QPiecewiseDoubleSpinBox::mousePressEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mousePressEvent(event);

	}

	void QPiecewiseDoubleSpinBox::mouseReleaseEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseReleaseEvent(event);
	}

	void QPiecewiseDoubleSpinBox::mouseMoveEvent(QMouseEvent* event)
	{
		QDoubleSpinBox::mouseMoveEvent(event);

	}

	void QPiecewiseDoubleSpinBox::ModifyValue(double v)
	{
		this->setRealValue(v);

	}

	void QPiecewiseDoubleSpinBox::ModifyValueAndUpdate(double v)
	{

		this->setKeyboardTracking(true);
		this->setRealValue(v);
		this->lineEdit()->setText(QString::number(realValue, 10, displayDecimals));
		this->setKeyboardTracking(false);
	}


}

