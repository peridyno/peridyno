#include "QMatrix4FieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"

namespace dyno
{
	//IMPL_FIELD_WIDGET(Mat4f, QMatrix4FieldWidget)
	int QMatrix4FieldWidget::reg_field_widget = []() {
		dyno::PPropertyWidget::registerWidget({ &typeid(Mat4f), &QMatrix4FieldWidget::createWidget });
		dyno::PPropertyWidget::registerWidget({ &typeid(Mat4d), &QMatrix4FieldWidget::createWidget });
		return 0;
	}();

	QWidget* QMatrix4FieldWidget::createWidget(dyno::FBase* f) {
		return new QMatrix4FieldWidget(f);
	}

	QMatrix4FieldWidget::QMatrix4FieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);//Label
		QToggleLabel* name = new QToggleLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);
		for (int i = 0; i < 16; ++i) {
			spinners[i] = new QPiecewiseDoubleSpinBox;
			spinners[i]->setMinimumWidth(30);
			spinners[i]->setRange(field->getMin(), field->getMax());
		}

		// Add label and spinners to layout
		layout->addWidget(name, 0, 0);
		
		// Matrix elements arranged in rows
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				layout->addWidget(spinners[i*4 + j], i+1, j+1);
			}
		}
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();
		
		if (template_name == std::string(typeid(Mat4f).name()))
		{
			FVar<Mat4f>* f = TypeInfo::cast<FVar<Mat4f>>(field);
			auto mat = f->getValue();
			
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					spinners[i*4 + j]->setRealValue(mat(i, j));
				}
			}
		}
		else if (template_name == std::string(typeid(Mat4d).name()))
		{
			FVar<Mat4d>* f = TypeInfo::cast<FVar<Mat4d>>(field);
			auto mat = f->getValue();
			
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					spinners[i*4 + j]->setRealValue(mat(i, j));
				}
			}
			
			for (int i = 0; i < 16; ++i) {
				spinners[i]->setDouble(true);
			}
		}

		for (int i = 0; i < 16; ++i) {
			QObject::connect(spinners[i], SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		}

		// Connect toggle signal for decimal precision
		for (int i = 0; i < 16; ++i) {
			QObject::connect(name, SIGNAL(toggle(bool)), spinners[i], SLOT(toggleDecimals(bool)));
		}

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	QMatrix4FieldWidget::~QMatrix4FieldWidget()
	{
	}

	void QMatrix4FieldWidget::updateField(double)
	{
		std::string template_name = field()->getTemplateName();
		
		if (template_name == std::string(typeid(Mat4f).name()))
		{
			FVar<Mat4f>* f = TypeInfo::cast<FVar<Mat4f>>(field());
			Mat4f mat;
			
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					mat(i, j) = (float)spinners[i*4 + j]->getRealValue();
				}
			}
			
			f->setValue(mat, false);
		}
		else if (template_name == std::string(typeid(Mat4d).name()))
		{
			FVar<Mat4d>* f = TypeInfo::cast<FVar<Mat4d>>(field());
			Mat4d mat;
			
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					mat(i, j) = spinners[i*4 + j]->getRealValue();
				}
			}
			
			f->setValue(mat, false);
		}

		emit fieldChanged();
	}

	void QMatrix4FieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();
		
		if (template_name == std::string(typeid(Mat4f).name()))
		{
			FVar<Mat4f>* f = TypeInfo::cast<FVar<Mat4f>>(field());
			auto mat = f->getValue();
			
			for (int i = 0; i < 16; ++i) {
				spinners[i]->blockSignals(true);
			}
			
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					spinners[i*4 + j]->setRealValue(mat(i, j));
				}
			}
			
			for (int i = 0; i < 16; ++i) {
				spinners[i]->blockSignals(false);
			}
		}
		else if (template_name == std::string(typeid(Mat4d).name()))
		{
			FVar<Mat4d>* f = TypeInfo::cast<FVar<Mat4d>>(field());
			auto mat = f->getValue();
			
			for (int i = 0; i < 16; ++i) {
				spinners[i]->blockSignals(true);
			}
			
			for (int i = 0; i < 4; ++i) {
				for (int j = 0; j < 4; ++j) {
					spinners[i*4 + j]->setRealValue(mat(i, j));
				}
			}
			
			for (int i = 0; i < 16; ++i) {
				spinners[i]->blockSignals(false);
			}
		}
	}
}
