#include "QTransform3FieldWidget.h"

#include <QGridLayout>

#include "Field.h"
#include "QPiecewiseDoubleSpinBox.h"
#include "Quat.h"

namespace dyno
{
	//IMPL_FIELD_WIDGET(Transform3f, QTransform3FieldWidget)

	int QTransform3FieldWidget::reg_field_widget = []() {
		dyno::PPropertyWidget::registerWidget({ &typeid(Transform3f), &QTransform3FieldWidget::createWidget });
		dyno::PPropertyWidget::registerWidget({ &typeid(Transform3d), &QTransform3FieldWidget::createWidget });
		return 0;
	}();

	QWidget* QTransform3FieldWidget::createWidget(dyno::FBase* f) {
			return new QTransform3FieldWidget(f);
	}

	QTransform3FieldWidget::QTransform3FieldWidget(FBase* field)
		: QFieldWidget(field), useAngleMode(true)
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

		for (int i = 0; i < 3; ++i) {
			transSpinners[i] = new QPiecewiseDoubleSpinBox;
			transSpinners[i]->setMinimumWidth(30);
			transSpinners[i]->setRange(field->getMin(), field->getMax());
		}

		for (int i = 0; i < 3; ++i) {
			scaleSpinners[i] = new QPiecewiseDoubleSpinBox;
			scaleSpinners[i]->setMinimumWidth(30);
			scaleSpinners[i]->setRange(field->getMin(), field->getMax());
		}

		for (int i = 0; i < 9; ++i) {
			rotSpinners[i] = new QPiecewiseDoubleSpinBox;
			rotSpinners[i]->setMinimumWidth(30);
			rotSpinners[i]->setRange(field->getMin(), field->getMax());
		}

		for (int i = 0; i < 3; ++i) {
			rotAngleSpinners[i] = new QPiecewiseDoubleSpinBox;
			rotAngleSpinners[i]->setMinimumWidth(30);
			rotAngleSpinners[i]->setRange(-360, 360); // Angle range in degrees
		}

		rotModeToggle = new QPushButton("A");
		rotModeToggle->setMinimumWidth(30);
		rotModeToggle->setStyleSheet(
			"QPushButton {"
			"    background-color: #3e3e3d;"
			"    color: white;"
			"}"
			"QPushButton:hover {"
			"    background-color: #616161;"
			"}"
			"QPushButton:pressed {"
			"    background-color: #000000;"
			"}"
		);

		layout->addWidget(name, 0, 0);
		
		// Translation 
		QLabel* transLabel = new QLabel("Translation:");
		transLabel->setFixedSize(100, 18);
		layout->addWidget(transLabel, 1, 0);
		for (int i = 0; i < 3; ++i) {
			layout->addWidget(transSpinners[i], 1, i+1);
		}
		
		QWidget* spacer1 = new QWidget();
		spacer1->setFixedHeight(5);
		layout->addWidget(spacer1, 2, 0, 1, 4);
		
		// Scale
		QLabel* scaleLabel = new QLabel("Scale:");
		scaleLabel->setFixedSize(100, 18);
		layout->addWidget(scaleLabel, 3, 0);
		for (int i = 0; i < 3; ++i) {
			layout->addWidget(scaleSpinners[i], 3, i+1);
		}
		
		QWidget* spacer2 = new QWidget();
		spacer2->setFixedHeight(5);
		layout->addWidget(spacer2, 4, 0, 1, 4);
		
		// Rotation 
		QLabel* rotLabel = new QLabel("Rotation:");
		rotLabel->setFixedSize(100, 18);
		QHBoxLayout* rotLabelLayout = new QHBoxLayout;
		rotLabelLayout->addWidget(rotLabel);
		rotLabelLayout->addWidget(rotModeToggle);

		layout->addLayout(rotLabelLayout, 5, 0);
		layout->setColumnStretch(1, 1);
		layout->setColumnStretch(2, 1);
		layout->setColumnStretch(3, 1);
		
		for (int i = 0; i < 3; ++i) {
			layout->addWidget(rotAngleSpinners[i], 5, i+1);
		}
				
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				layout->addWidget(rotSpinners[i*3 + j], i+5, j+1);
			}
		}
		
		for (int i = 0; i < 9; ++i) {
			rotSpinners[i]->hide();
		}
		for (int i = 0; i < 3; ++i) {
			rotAngleSpinners[i]->show();
		}
		
		layout->setSpacing(3);

		std::string template_name = field->getTemplateName();
		
		if (template_name == std::string(typeid(Transform3f).name()))
		{
			FVar<Transform3f>* f = TypeInfo::cast<FVar<Transform3f>>(field);
			auto transform = f->getValue();		
			// Translation
			auto trans = transform.translation();
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->setRealValue(trans[i]);
			}	
			// Scale
			auto scale = transform.scale();
			for (int i = 0; i < 3; ++i) {
				scaleSpinners[i]->setRealValue(scale[i]);
			}		
			// Rotation
			auto rot = transform.rotation();
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					rotSpinners[i*3 + j]->setRealValue(rot(i, j));
				}
			}
			
			// Calculate rotation
			Quat<float> quat(rot);
			Vec3f angles;
			quat.toEulerAngle(angles.y, angles.x, angles.z);

			for (int i = 0; i < 3; ++i) {
				rotAngleSpinners[i]->setRealValue(angles[i] * 180.0f / M_PI); // Convert to degrees
			}
		}
		else if (template_name == std::string(typeid(Transform3d).name()))
		{
			FVar<Transform3d>* f = TypeInfo::cast<FVar<Transform3d>>(field);
			auto transform = f->getValue();		
			// Translation
			auto trans = transform.translation();
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->setRealValue(trans[i]);
			}	
			// Scale
			auto scale = transform.scale();
			for (int i = 0; i < 3; ++i) {
				scaleSpinners[i]->setRealValue(scale[i]);
			}	
			// Rotation
			auto rot = transform.rotation();
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					rotSpinners[i*3 + j]->setRealValue(rot(i, j));
				}
			}		
			// Calculate rotation 
			Quat<double> quat(rot);
			Vec3d angles;
			quat.toEulerAngle(angles.y, angles.z, angles.x);
			for (int i = 0; i < 3; ++i) {
				rotAngleSpinners[i]->setRealValue(angles[i] * 180.0 / M_PI); // Convert to degrees
			}	
			// Set to double precision
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->setDouble(true);
				scaleSpinners[i]->setDouble(true);
				rotAngleSpinners[i]->setDouble(true);
			}
			for (int i = 0; i < 9; ++i) {
				rotSpinners[i]->setDouble(true);
			}
		}

		// Connect signals
		for (int i = 0; i < 3; ++i) {
			QObject::connect(transSpinners[i], SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
			QObject::connect(scaleSpinners[i], SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
			QObject::connect(rotAngleSpinners[i], SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		}
		for (int i = 0; i < 9; ++i) {
			QObject::connect(rotSpinners[i], SIGNAL(editingFinishedWithValue(double)), this, SLOT(updateField(double)));
		}

		// Connect rotation mode toggle
		QObject::connect(rotModeToggle, &QPushButton::clicked, [this]() {
			useAngleMode = !useAngleMode;
			
			if (useAngleMode) {
				rotModeToggle->setText("A");

				for (auto spinner : rotSpinners) {
					if (spinner) spinner->hide();
				}
				for (auto spinner : rotAngleSpinners) {
					if (spinner) spinner->show();
				}
			}
			else {
				rotModeToggle->setText("M");

				for (auto spinner : rotAngleSpinners) {
					if (spinner) spinner->hide();
				}
				for (auto spinner : rotSpinners) {
					if (spinner) spinner->show();
				}
			}
		});

		// Connect toggle signal for decimal precision
		for (int i = 0; i < 3; ++i) {
			QObject::connect(name, SIGNAL(toggle(bool)), transSpinners[i], SLOT(toggleDecimals(bool)));
			QObject::connect(name, SIGNAL(toggle(bool)), scaleSpinners[i], SLOT(toggleDecimals(bool)));
			QObject::connect(name, SIGNAL(toggle(bool)), rotAngleSpinners[i], SLOT(toggleDecimals(bool)));
		}
		for (int i = 0; i < 9; ++i) {
			QObject::connect(name, SIGNAL(toggle(bool)), rotSpinners[i], SLOT(toggleDecimals(bool)));
		}

		QObject::connect(this, SIGNAL(fieldChanged()), this, SLOT(updateWidget()));
	}

	QTransform3FieldWidget::~QTransform3FieldWidget()
	{
	}

	void QTransform3FieldWidget::updateField(double)
	{
		std::string template_name = field()->getTemplateName();
		
		if (template_name == std::string(typeid(Transform3f).name()))
		{
			FVar<Transform3f>* f = TypeInfo::cast<FVar<Transform3f>>(field());
			Transform3f transform;
			
			// Translation
			Vec3f trans;
			for (int i = 0; i < 3; ++i) {
				trans[i] = (float)transSpinners[i]->getRealValue();
			}
			transform.translation() = trans;
			
			// Scale
			Vec3f scale;
			for (int i = 0; i < 3; ++i) {
				scale[i] = (float)scaleSpinners[i]->getRealValue();
			}
			transform.scale() = scale;
			
			// Rotation
			if (useAngleMode) {
				// Use angle input
				Vec3f angles;
				for (int i = 0; i < 3; ++i) {
					angles[i] = (float)rotAngleSpinners[i]->getRealValue() * M_PI / 180.0f; // Convert to radians
				}
				
				Quat<float> quat(angles.y, angles.x, angles.z);
				Mat3f rot = quat.toMatrix3x3();
				transform.rotation() = rot;
			} else {
				// Use matrix input
				Mat3f rot;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						rot(i, j) = (float)rotSpinners[i*3 + j]->getRealValue();
					}
				}
				transform.rotation() = rot;
			}
			
			f->setValue(transform, false);
		}
		else if (template_name == std::string(typeid(Transform3d).name()))
		{
			FVar<Transform3d>* f = TypeInfo::cast<FVar<Transform3d>>(field());
			Transform3d transform;
			
			// Translation
			Vec3d trans;
			for (int i = 0; i < 3; ++i) {
				trans[i] = transSpinners[i]->getRealValue();
			}
			transform.translation() = trans;
			
			// Scale
			Vec3d scale;
			for (int i = 0; i < 3; ++i) {
				scale[i] = scaleSpinners[i]->getRealValue();
			}
			transform.scale() = scale;
			
			// Rotation
			if (useAngleMode) {
				// Use angle input
				Vec3d angles;
				for (int i = 0; i < 3; ++i) {
					angles[i] = rotAngleSpinners[i]->getRealValue() * M_PI / 180.0; // Convert to radians
				}
				
				Quat<double> quat(angles[0], angles[1], angles[2]);			
				Mat3d rot = quat.toMatrix3x3();
				transform.rotation() = rot;
			} else {
				// Use matrix input
				Mat3d rot;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						rot(i, j) = rotSpinners[i*3 + j]->getRealValue();
					}
				}
				transform.rotation() = rot;
			}
			
			f->setValue(transform, false);
		}

		emit fieldChanged();
	}

	void QTransform3FieldWidget::updateWidget()
	{
		std::string template_name = field()->getTemplateName();
		
		if (template_name == std::string(typeid(Transform3f).name()))
		{
			FVar<Transform3f>* f = TypeInfo::cast<FVar<Transform3f>>(field());
			auto transform = f->getValue();
			
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->blockSignals(true);
				scaleSpinners[i]->blockSignals(true);
				rotAngleSpinners[i]->blockSignals(true);
			}
			for (int i = 0; i < 9; ++i) {
				rotSpinners[i]->blockSignals(true);
			}
			
			// Translation
			auto trans = transform.translation();
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->setRealValue(trans[i]);
			}
			
			// Scale
			auto scale = transform.scale();
			for (int i = 0; i < 3; ++i) {
				scaleSpinners[i]->setRealValue(scale[i]);
			}
			
			// Rotation
			auto rot = transform.rotation();
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					rotSpinners[i*3 + j]->setRealValue(rot(i, j));
				}
			}
			
			Quat<float> quat(rot);
			Vec3f angles;
			quat.toEulerAngle(angles.y, angles.x, angles.z);
			for (int i = 0; i < 3; ++i) {
				rotAngleSpinners[i]->setRealValue(angles[i] * 180.0f / M_PI); // Convert to degrees
			}
			
			// Unblock signals
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->blockSignals(false);
				scaleSpinners[i]->blockSignals(false);
				rotAngleSpinners[i]->blockSignals(false);
			}
			for (int i = 0; i < 9; ++i) {
				rotSpinners[i]->blockSignals(false);
			}
		}
		else if (template_name == std::string(typeid(Transform3d).name()))
		{
			FVar<Transform3d>* f = TypeInfo::cast<FVar<Transform3d>>(field());
			auto transform = f->getValue();
			
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->blockSignals(true);
				scaleSpinners[i]->blockSignals(true);
				rotAngleSpinners[i]->blockSignals(true);
			}
			for (int i = 0; i < 9; ++i) {
				rotSpinners[i]->blockSignals(true);
			}
			
			// Translation
			auto trans = transform.translation();
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->setRealValue(trans[i]);
			}
			
			// Scale
			auto scale = transform.scale();
			for (int i = 0; i < 3; ++i) {
				scaleSpinners[i]->setRealValue(scale[i]);
			}
			
			// Rotation
			auto rot = transform.rotation();
			
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					rotSpinners[i*3 + j]->setRealValue(rot(i, j));
				}
			}
			
			Quat<double> quat(rot);
			Vec3d angles;
			quat.toEulerAngle(angles.y, angles.x, angles.z);
			for (int i = 0; i < 3; ++i) {
				rotAngleSpinners[i]->setRealValue(angles[i] * 180.0 / M_PI); // Convert to degrees
			}
			
			for (int i = 0; i < 3; ++i) {
				transSpinners[i]->blockSignals(false);
				scaleSpinners[i]->blockSignals(false);
				rotAngleSpinners[i]->blockSignals(false);
			}
			for (int i = 0; i < 9; ++i) {
				rotSpinners[i]->blockSignals(false);
			}
		}
	}
}