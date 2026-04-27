#include "QMatrix3FieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"

namespace dyno
{
	//IMPL_FIELD_WIDGET(Mat3f, QMatrix3FieldWidget)

	int QMatrix3FieldWidget::reg_field_widget = []() {
		dyno::PPropertyWidget::registerWidget({ &typeid(Mat3f), &QMatrix3FieldWidget::createWidget });
		dyno::PPropertyWidget::registerWidget({ &typeid(Mat3d), &QMatrix3FieldWidget::createWidget });
		return 0;
	}();

	QWidget* QMatrix3FieldWidget::createWidget(dyno::FBase* f) {
			return new QMatrix3FieldWidget(f);
	}

	QMatrix3FieldWidget::QMatrix3FieldWidget(FBase* field)
		: QFieldWidget(field)
	{
		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		this->setLayout(layout);

		//Label
		QToggleLabel* name = new QToggleLabel();
		QString str = FormatFieldWidgetName(field->getObjectName());
		name->setFixedSize(100, 18);
		QFontMetrics fontMetrics(name->font());
		QString elide = fontMetrics.elidedText(str, Qt::ElideRight, 100);
		name->setText(elide);
		//Set label tips
		name->setToolTip(str);

		for (int i = 0; i < 9; ++i) {
			spinners[i] = new QPiecewiseDoubleSpinBox;
			spinners[i]->setMinimumWidth(30);
			spinners[i]->setRange(field->getMin(), field->getMax());
		}

		layout->addWidget(name, 0, 0);
		
		// Matrix elements
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				layout->addWidget(spinners[i*3 + j], i+1, j+1);
			}
		}
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();
		
		if (template_name == std::string(typeid(Mat3f).name()))
		{
			FVar<Mat3f>* f = TypeInfo::cast<FVar<Mat3f>>(field);
			auto mat = f->getValue();
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					spinners[i*3 + j]->setRealValue(mat(i, j));
				}
			}
		}
		else if (template_name == std::string(typeid(Mat3d).name()))
		{
			FVar<Mat3d>* f = TypeInfo::cast<FVar<Mat3d>>(field);
			auto mat = f->getValue();
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					spinners[i*3 + j]->setRealValue(mat(i, j));
				}
			}
			
			for (int i = 0; i < 9; ++i) {
				spinners[i]->setDouble(true);
			}
		}

		for (int i = 0; i < 9; ++i) {
			QObject::connect(spinners[i], SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		}

		// Connect toggle signal for decimal precision
		for (int i = 0; i < 9; ++i) {
			QObject::connect(name, SIGNAL(toggle(bool)), spinners[i], SLOT(toggleDecimals(bool)));
		}

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	QMatrix3FieldWidget::~QMatrix3FieldWidget()
	{
	}

	void QMatrix3FieldWidget::updateField(double)
	{
		std::string template_name = field()->getTemplateName();
		
		if (template_name == std::string(typeid(Mat3f).name()))
		{
			FVar<Mat3f>* f = TypeInfo::cast<FVar<Mat3f>>(field());
			Mat3f mat;
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					mat(i, j) = (float)spinners[i*3 + j]->getRealValue();
				}
			}
			
			f->setValue(mat, false);
		}
		else if (template_name == std::string(typeid(Mat3d).name()))
		{
			FVar<Mat3d>* f = TypeInfo::cast<FVar<Mat3d>>(field());
			Mat3d mat;
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					mat(i, j) = spinners[i*3 + j]->getRealValue();
				}
			}
			
			f->setValue(mat, false);
		}

		emit fieldChanged();
	}

	void QMatrix3FieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();
		
		if (template_name == std::string(typeid(Mat3f).name()))
		{
			FVar<Mat3f>* f = TypeInfo::cast<FVar<Mat3f>>(field());
			auto mat = f->getValue();
			
			for (int i = 0; i < 9; ++i) {
				spinners[i]->blockSignals(true);
			}
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					spinners[i*3 + j]->setRealValue(mat(i, j));
				}
			}
			
			for (int i = 0; i < 9; ++i) {
				spinners[i]->blockSignals(false);
			}
		}
		else if (template_name == std::string(typeid(Mat3d).name()))
		{
			FVar<Mat3d>* f = TypeInfo::cast<FVar<Mat3d>>(field());
			auto mat = f->getValue();
			
			for (int i = 0; i < 9; ++i) {
				spinners[i]->blockSignals(true);
			}
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					spinners[i*3 + j]->setRealValue(mat(i, j));
				}
			}
			
			for (int i = 0; i < 9; ++i) {
				spinners[i]->blockSignals(false);
			}
		}
	}
}
