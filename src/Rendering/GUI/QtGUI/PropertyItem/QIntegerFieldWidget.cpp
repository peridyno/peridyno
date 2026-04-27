#include "QIntegerFieldWidget.h"

#include <QGridLayout>
#include <QDialog>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QPushButton>
#include <QMenu>
#include <QMouseEvent>
#include <climits>

#include "Field.h"
#include "QPiecewiseSpinBox.h"

namespace dyno
{
	IMPL_FIELD_WIDGET(int, QIntegerFieldWidget)
	IMPL_FIELD_WIDGET(uint, QUIntegerFieldWidget)

	QIntegerFieldWidget::QIntegerFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<int>* f = TypeInfo::cast<FVar<int>>(field);
		if (f == nullptr) {
			return;
		}

		//this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		mSpinner = new QPiecewiseSpinBox;
		mSpinner->setRange(castMinimum<int>(field->getMin()), castMaximum<int>(field->getMax()));
		mSpinner->setValue(f->getValue());
		mSpinner->setFixedWidth(100);

		layout->addWidget(name, 0, 0);
		layout->addWidget(mSpinner, 0, 1, Qt::AlignRight);

		this->connect(mSpinner, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));

		mRightMenu = new QMenu(this);
		mRightMenu->addAction("Set Range", this, &QIntegerFieldWidget::editRange);

		this->installEventFilter(this);

	}

	void QIntegerFieldWidget::updateField(int value)
	{

		FVar<int>* f = TypeInfo::cast<FVar<int>>(field());
		if (f == nullptr)
			return;

		f->setValue(value,false);
		emit fieldChanged();

	}

	QUIntegerFieldWidget::QUIntegerFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(field);
		if (f == nullptr)
		{
			return;
		}

		//this->setStyleSheet("border:none");
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QLabel* name = new QLabel();
		name->setFixedHeight(18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		mSpinner = new QPiecewiseSpinBox;
		mSpinner->setFixedWidth(100);

		mSpinner->setRange(castMinimum<int>(field->getMin()) >= 0 ? castMinimum<int>(field->getMin()) : 0,
			castMaximum<int>(field->getMax()) >= 0 ? castMaximum<int>(field->getMax()) : 0);
		mSpinner->setValue(f->getValue());

		layout->addWidget(name, 0, 0);
		layout->addWidget(mSpinner, 0, 1, Qt::AlignRight);
		layout->setSpacing(3);

		this->connect(mSpinner, SIGNAL(valueChanged(int)), this, SLOT(updateField(int)));

		mRightMenu = new QMenu(this);
		mRightMenu->addAction("Set Range", this, &QUIntegerFieldWidget::editRange);

		this->installEventFilter(this);
	}

	void QUIntegerFieldWidget::updateField(int value)
	{
		FVar<uint>* f = TypeInfo::cast<FVar<uint>>(field());
		if (f == nullptr)
			return;

		f->setValue(value,false);
		emit fieldChanged();
	}

	bool QIntegerFieldWidget::eventFilter(QObject* watched, QEvent* event)
	{
		if (watched == this && event->type() == QEvent::MouseButtonPress)
		{
			QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
			if (mouseEvent->button() == Qt::RightButton)
			{
				mRightMenu->exec(mouseEvent->globalPos());
				return true;
			}
		}
		return false;
	}

	void QIntegerFieldWidget::editRange()
	{
		QDialog* dialog = new QDialog(this);
		dialog->setWindowTitle("Set Range");
		dialog->setFixedSize(300, 150);

		QVBoxLayout* mainLayout = new QVBoxLayout(dialog);
		QFormLayout* formLayout = new QFormLayout();
		QSpinBox* minSpinBox = new QSpinBox();
		QSpinBox* maxSpinBox = new QSpinBox();

		minSpinBox->setRange(INT_MIN, INT_MAX);
		maxSpinBox->setRange(INT_MIN, INT_MAX);
		minSpinBox->setValue(castMinimum<int>(field()->getMin()));
		maxSpinBox->setValue(castMaximum<int>(field()->getMax()));
		formLayout->addRow("Minimum:", minSpinBox);
		formLayout->addRow("Maximum:", maxSpinBox);

		QHBoxLayout* buttonLayout = new QHBoxLayout();
		QPushButton* okButton = new QPushButton("OK");
		QPushButton* cancelButton = new QPushButton("Cancel");

		buttonLayout->addStretch();
		buttonLayout->addWidget(okButton);
		buttonLayout->addWidget(cancelButton);
		mainLayout->addLayout(formLayout);
		mainLayout->addLayout(buttonLayout);

		QObject::connect(okButton, &QPushButton::clicked, [=]() {
			int min = minSpinBox->value();
			int max = maxSpinBox->value();

			if (min > max) {
				std::swap(min, max);
				minSpinBox->setValue(min);
				maxSpinBox->setValue(max);
				return;
			}

			field()->setRange(min, max);
			mSpinner->setRange(min, max);
			dialog->accept();
		});

		QObject::connect(cancelButton, &QPushButton::clicked, dialog, &QDialog::reject);
		dialog->exec();		
		delete dialog;
	}

	bool QUIntegerFieldWidget::eventFilter(QObject* watched, QEvent* event)
	{
		if (watched == this && event->type() == QEvent::MouseButtonPress)
		{
			QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
			if (mouseEvent->button() == Qt::RightButton)
			{
				mRightMenu->exec(mouseEvent->globalPos());
				return true;
			}
		}
		return false;
	}

	void QUIntegerFieldWidget::editRange()
	{
		QDialog* dialog = new QDialog(this);
		dialog->setWindowTitle("Set Range");
		dialog->setFixedSize(300, 150);

		QVBoxLayout* mainLayout = new QVBoxLayout(dialog);
		QFormLayout* formLayout = new QFormLayout();
		QSpinBox* minSpinBox = new QSpinBox();
		QSpinBox* maxSpinBox = new QSpinBox();

		int currentMin = castMinimum<int>(field()->getMin()) >= 0 ? castMinimum<int>(field()->getMin()) : 0;
		int currentMax = castMaximum<int>(field()->getMax()) >= 0 ? castMaximum<int>(field()->getMax()) : 0;
		
		minSpinBox->setRange(0, INT_MAX);
		maxSpinBox->setRange(0, INT_MAX);

		minSpinBox->setValue(currentMin);
		maxSpinBox->setValue(currentMax);

		formLayout->addRow("Minimum:", minSpinBox);
		formLayout->addRow("Maximum:", maxSpinBox);

		QHBoxLayout* buttonLayout = new QHBoxLayout();
		QPushButton* okButton = new QPushButton("OK");
		QPushButton* cancelButton = new QPushButton("Cancel");

		buttonLayout->addStretch();
		buttonLayout->addWidget(okButton);
		buttonLayout->addWidget(cancelButton);

		mainLayout->addLayout(formLayout);
		mainLayout->addLayout(buttonLayout);

		QObject::connect(okButton, &QPushButton::clicked, [=]() {
			int min = minSpinBox->value();
			int max = maxSpinBox->value();

			if (min > max) {
				std::swap(min, max);
				minSpinBox->setValue(min);
				maxSpinBox->setValue(max);
				return;
			}

			field()->setRange(min, max);
			mSpinner->setRange(min, max);
			dialog->accept();
		});

		QObject::connect(cancelButton, &QPushButton::clicked, dialog, &QDialog::reject);

		dialog->exec();
		delete dialog;
	}
}