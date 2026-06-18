#include "QRealFieldWidget.h"

#include <QHBoxLayout>
#include <QDialog>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QPushButton>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"
#include <QMenu>
#include <QMouseEvent>

namespace dyno
{
	//IMPL_FIELD_WIDGET(float, QRealFieldWidget)

	int QRealFieldWidget::reg_field_widget = []()
		{
			dyno::PPropertyWidget::registerWidget({ &typeid(float), &QRealFieldWidget::createWidget });
			dyno::PPropertyWidget::registerWidget({ &typeid(double), &QRealFieldWidget::createWidget });
			return 0;
		}();

		QWidget* QRealFieldWidget::createWidget(dyno::FBase* f) {
			return new QRealFieldWidget(f);
		}

		QRealFieldWidget::QRealFieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		//this->setStyleSheet("border:none");
		QHBoxLayout* layout = new QHBoxLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		QToggleLabel* name = new QToggleLabel();
		name->setFixedHeight(24);
		name->setText(FormatFieldWidgetName(field->getObjectName()));

		slider = new QDoubleSlider;
		slider->setRange(field->getMin(), field->getMax());
		slider->setMinimumWidth(60);

		spinner = new QPiecewiseDoubleSpinBox();
		spinner->setRange(field->getMin(), field->getMax());
		spinner->setFixedWidth(100);

		std::string template_name = field->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field);

			spinner->setRealValue((double)f->getValue());
			slider->setValue((double)f->getValue());
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field);

			spinner->setRealValue((double)f->getValue());
			spinner->setDouble(true);
			slider->setValue(f->getValue());
		}

		layout->addWidget(name, 0);
		layout->addWidget(slider, 1);
		layout->addStretch();
		layout->addWidget(spinner, 2);
		layout->setSpacing(3);

		FormatFieldWidgetName(field->getObjectName());

		QObject::connect(slider, SIGNAL(valueChanged(double)), this, SLOT(onSliderValueChanged(double)));

		QObject::connect(spinner, SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));

		QObject::connect(name, SIGNAL(toggle(bool)), spinner, SLOT(toggleDecimals(bool)));

		mRightMenu = new QMenu(this);
		mRightMenu->addAction("Set Range", this, &QRealFieldWidget::editRange);

		this->installEventFilter(this);

	}

	QRealFieldWidget::~QRealFieldWidget()
	{
	}

	void QRealFieldWidget::updateField(double value)
	{
		std::string template_name = field()->getTemplateName();

		double v = spinner->getRealValue();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field());
			f->setValue((float)v, false);
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field());
			f->setValue(v, false);
		}

		emit fieldChanged();
	}

	void QRealFieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();
		if (template_name == std::string(typeid(float).name()))
		{
			FVar<float>* f = TypeInfo::cast<FVar<float>>(field());

			slider->blockSignals(true);
			slider->setValue((double)f->getValue());
			slider->blockSignals(false);
		}
		else if (template_name == std::string(typeid(double).name()))
		{
			FVar<double>* f = TypeInfo::cast<FVar<double>>(field());

			slider->blockSignals(true);
			slider->setValue(f->getValue());
			slider->blockSignals(false);
		}
	}

	void QRealFieldWidget::onSliderValueChanged(double val)
	{
		spinner->triggerEditingFinished(val);
	}


	bool QRealFieldWidget::eventFilter(QObject* watched, QEvent* event)
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

	void QRealFieldWidget::editRange() 
	{
		// Create a dialog for setting range
		QDialog* dialog = new QDialog(this);
		dialog->setWindowTitle("Set Range");
		dialog->setFixedSize(300, 150);

		// Create layout
		QVBoxLayout* mainLayout = new QVBoxLayout(dialog);
		QFormLayout* formLayout = new QFormLayout();

		// Create spin boxes for min and max values
		QDoubleSpinBox* minSpinBox = new QDoubleSpinBox();
		QDoubleSpinBox* maxSpinBox = new QDoubleSpinBox();

		minSpinBox->setRange(-1e9, 1e9);
		maxSpinBox->setRange(-1e9, 1e9);

		minSpinBox->setValue(field()->getMin());
		maxSpinBox->setValue(field()->getMax());

		formLayout->addRow("Minimum:", minSpinBox);
		formLayout->addRow("Maximum:", maxSpinBox);

		QHBoxLayout* buttonLayout = new QHBoxLayout();
		QPushButton* okButton = new QPushButton("OK");
		QPushButton* cancelButton = new QPushButton("Cancel");

		const QString btnStyle = R"(
			QPushButton {
				background-color: #464646;
				border-radius: 4px;
			}
			QPushButton:hover {
				background-color: #616161;
			}
			QPushButton:pressed {
				background-color: #000000;
			}
		)";

		okButton->setStyleSheet(btnStyle);
		cancelButton->setStyleSheet(btnStyle);

		buttonLayout->addStretch();
		buttonLayout->addWidget(okButton);
		buttonLayout->addWidget(cancelButton);

		mainLayout->addLayout(formLayout);
		mainLayout->addLayout(buttonLayout);

		QObject::connect(okButton, &QPushButton::clicked, [=]() {
			double min = minSpinBox->value();
			double max = maxSpinBox->value();

			if (min > max) {
				std::swap(min, max);
				minSpinBox->setValue(min);
				maxSpinBox->setValue(max);
				return;
			}

			field()->setRange(min, max);
			slider->setRange(min, max);
			spinner->setRange(min, max);
			dialog->accept();
		});

		QObject::connect(cancelButton, &QPushButton::clicked, dialog, &QDialog::reject);
		dialog->exec();
		delete dialog;
	}
}

