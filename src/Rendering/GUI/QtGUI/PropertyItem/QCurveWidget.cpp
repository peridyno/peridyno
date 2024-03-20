#include "QCurveWidget.h"

#include <QComboBox>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QPainter>
#include <QSpacerItem>
#include <QPainterPath>
#include "PCustomWidgets.h"
#include <QCheckBox>
#include "QToggleButton.h"

#include "Field.h"

namespace dyno
{
	IMPL_FIELD_WIDGET(Curve, QCurveWidget)

	QCurveWidget::QCurveWidget(FBase* field)
		: QFieldWidget(field)
	{
		printf("QCurveWidget\n");
		FVar<Curve>* f = TypeInfo::cast<FVar<Curve>>(field);
		if (f == nullptr)
		{
			printf("QCurve Nullptr\n");
			return;

		}



		

		//Enum List    Direction Mode

		//Enum List    InterpMode
		int curIndex2 = int(f->getValue().InterpMode);
		int enumNum2 = f->getValue().InterpolationCount;
		QComboBox* combox2 = new QComboBox;
		combox2->setMaximumWidth(256);
			for (size_t i = 0; i < 2; i++)
			{
				auto enumName2 = f->getValue().InterpStrings[i];
				combox2->addItem(QString::fromStdString(enumName2));
			}
		combox2->setCurrentIndex(curIndex2);
		combox2->setStyleSheet("background-color: qlineargradient(spread : pad, x1 : 0, y1 : 0, x2 : 0, y2 : 0.7, stop : 0 rgba(100, 100, 100, 255), stop : 1 rgba(35, 35, 35, 255)); ");

		auto s = f->getValue();

		//build RampWidget by field
		//RampWidget name
		QLabel* name = new QLabel();
		name->setFixedSize(80, 18);
		name->setText(FormatFieldWidgetName(field->getObjectName()));
		name->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);//



		//build DrawLabel
		QCurveLabel* DrawLabel = new QCurveLabel(f);




		connect(combox2, SIGNAL(currentIndexChanged(int)), DrawLabel, SLOT(changeInterpValue(int)));

		//VLayout : combox2 / SquardButton / CloseButton
		QVBoxLayout* VLayout = new QVBoxLayout();
		QSpacerItem* VSpacer = new QSpacerItem(10, 380, QSizePolicy::Minimum, QSizePolicy::Expanding);
		VLayout->addItem(VSpacer);
		VLayout->addWidget(combox2);
		VLayout->setSpacing(5);


		QGridLayout* Gridlayout = new QGridLayout;
		Gridlayout->setContentsMargins(0, 0, 0, 0);
		Gridlayout->setSpacing(5);
		Gridlayout->addWidget(name, 0, 0,3,1, Qt::AlignLeft);
		Gridlayout->addWidget(DrawLabel,0,1, Qt::AlignCenter);
		Gridlayout->addLayout(VLayout,0,2, Qt::AlignRight);
		Gridlayout->setColumnStretch(0, 0);
		Gridlayout->setColumnStretch(1, 5);
		Gridlayout->setColumnStretch(2, 0);


		QLabel* spacingName = new QLabel("Spacing");

		mQDoubleSlider* spacingSlider = new mQDoubleSlider;
		spacingSlider->nameLabel = spacingName;
		spacingSlider->setRange(1, 40);
		spacingSlider->setValue(f->getValue().Spacing);
		spacingSlider->id = -1;
		mQDoubleSpinner* spacingSpinner = new mQDoubleSpinner;
		spacingSpinner->setRange(1, 40);
		spacingSpinner->setValue(f->getValue().Spacing);
		spacingSpinner->id = -1;
		QObject::connect(spacingSlider, SIGNAL(valueChanged(double)), spacingSpinner, SLOT(setValue(double)));
		QObject::connect(spacingSpinner, SIGNAL(valueChanged(double)), spacingSlider, SLOT(setValue(double)));
		QObject::connect(spacingSpinner, SIGNAL(valueChangedAndID(double, int)), DrawLabel, SLOT(setSpacingToDrawLabel(double,int)));
	
		QHBoxLayout* SpacingHlayout = new QHBoxLayout;
		SpacingHlayout->setContentsMargins(0, 0, 0, 0);
		SpacingHlayout->setSpacing(0);
		SpacingHlayout->addWidget(spacingName);
		SpacingHlayout->addWidget(spacingSlider);
		SpacingHlayout->addWidget(spacingSpinner);


		QLabel* nameReal[4];
		mQDoubleSlider* sliderReal[4];
		mQDoubleSpinner* spinnerReal[4];
		std::string nameString[4] = { "MinX","MinY","MaxX","MaxY" };

		QHBoxLayout* Hlayout1 = new QHBoxLayout;
		Hlayout1->setContentsMargins(0, 0, 0, 0);
		Hlayout1->setSpacing(0);

		QHBoxLayout* Hlayout2 = new QHBoxLayout;
		Hlayout2->setContentsMargins(0, 0, 0, 0);
		Hlayout2->setSpacing(0);

		for (size_t i = 0; i < 4; i++) 
		{
			nameReal[i] = new QLabel(QString::fromStdString(nameString[i]));

			sliderReal[i] = new mQDoubleSlider;
			sliderReal[i]->id = i;

			spinnerReal[i] = new mQDoubleSpinner;
			spinnerReal[i]->id = i;

			auto rangeArray = f->getValue().remapRange;

			sliderReal[i]->setRange(rangeArray[2 * i], rangeArray[2 * i + 1]);
			spinnerReal[i]->setRange(rangeArray[2 * i], rangeArray[2 * i + 1]);

			QObject::connect(sliderReal[i], SIGNAL(valueChanged(double)), spinnerReal[i], SLOT(setValue(double)));
			QObject::connect(spinnerReal[i], SIGNAL(valueChanged(double)), sliderReal[i], SLOT(setValue(double)));
			QObject::connect(spinnerReal[i], SIGNAL(valueChangedAndID(double, int)), DrawLabel, SLOT(SetValueToDrawLabel(double, int)));

			switch (i)
			{
				case 0: 
					sliderReal[i]->setValue(f->getValue().NminX);
					spinnerReal[i]->setValue(f->getValue().NminX);
					break;

				case 1: 
					sliderReal[i]->setValue(f->getValue().NminY);
					spinnerReal[i]->setValue(f->getValue().NminY);
					break;

				case 2: 
					sliderReal[i]->setValue(f->getValue().NmaxX);
					spinnerReal[i]->setValue(f->getValue().NmaxX);
					break;

				case 3: 
					sliderReal[i]->setValue(f->getValue().NmaxY);
					spinnerReal[i]->setValue(f->getValue().NmaxY);
					break;
			}



			if (i < 2) 
			{
				Hlayout1->addWidget(nameReal[i]);
				Hlayout1->addWidget(sliderReal[i]);
				Hlayout1->addWidget(spinnerReal[i]);
			}
			else 
			{
				Hlayout2->addWidget(nameReal[i]);
				Hlayout2->addWidget(sliderReal[i]);
				Hlayout2->addWidget(spinnerReal[i]);
			}
		}

		//Bool
		QHBoxLayout* boolLayout = new QHBoxLayout;
		boolLayout->setContentsMargins(0, 0, 0, 0);
		boolLayout->setSpacing(0);

		QLabel* boolName = new QLabel();
		boolName->setFixedHeight(24);
		boolName->setText("Resample");
		mQCheckBox* Checkbox = new mQCheckBox();
		Checkbox->nameLabel = boolName;
		Checkbox->QWidget::setFixedWidth(20); 
		Checkbox->QAbstractButton::setChecked(f->getValue().resample);

		connect(Checkbox, SIGNAL(mValueChanged(int)), DrawLabel, SLOT(setLinearResample(int)));
		connect(Checkbox, SIGNAL(mValueChanged(int)), spacingSlider, SLOT(setNewVisable(int)));
		connect(Checkbox, SIGNAL(mValueChanged(int)), spacingSpinner, SLOT(setNewVisable(int)));

		connect(combox2, SIGNAL(currentIndexChanged(int)), Checkbox, SLOT(updateChecked(int)));

		if (f->getValue().resample == false)
		{
			spacingSlider->setVisible(false);
			spacingSpinner->setVisible(false);
			spacingName->setVisible(false);
		}


		boolLayout->addWidget(boolName, 0);
		boolLayout->addStretch(1);
		boolLayout->addWidget(Checkbox, 0);

		//if (f->getDataPtr()->InterpMode == Ramp::Interpolation::Bezier) { boolName->setVisible(0); Checkbox->setVisible(0); }




		if (f->getValue().useSquardButton)
		{
			QToggleButton* unfold = new QToggleButton(f->getValue().useSquard);

			unfold->setText("Squard", "Rect");
			unfold->setStyleSheet(
				"QPushButton{background-color: qlineargradient(spread : pad, x1 : 0, y1 : 0, x2 : 0, y2 : 0.7, stop : 0 rgba(100, 100, 100, 255), stop : 1 rgba(35, 35, 35, 255));}QPushButton:hover{background-color: qlineargradient(spread : pad, x1 : 0, y1 : 0, x2 : 0, y2 : 0.7, stop : 0 rgba(120,120,120, 255), stop : 1 rgba(90, 90, 90, 255));} "
			);
			connect(unfold, &QToggleButton::clicked, DrawLabel, &QCurveLabel::changeLabelSize);
			unfold->setValue(f->getValue().useSquard);
			VLayout->addWidget(unfold);
		}

		
		{	
			if(f->getValue().useColseButton)
			{
				QToggleButton* curveCloseButton = new QToggleButton(f->getValue().curveClose);
				curveCloseButton->setText("Close", "Open");
				curveCloseButton->setStyleSheet
				(
					"QPushButton{background-color: qlineargradient(spread : pad, x1 : 0, y1 : 0, x2 : 0, y2 : 0.7, stop : 0 rgba(100, 100, 100, 255), stop : 1 rgba(35, 35, 35, 255));}QPushButton:hover{background-color: qlineargradient(spread : pad, x1 : 0, y1 : 0, x2 : 0, y2 : 0.7, stop : 0 rgba(120,120,120, 255), stop : 1 rgba(90, 90, 90, 255));} "
				);
				VLayout->addWidget(curveCloseButton); 
				connect(curveCloseButton, &QToggleButton::clicked, DrawLabel, &QCurveLabel::setCurveClose);
				curveCloseButton->setValue(f->getValue().curveClose);
			}
		}


		QVBoxLayout* TotalLayout = new QVBoxLayout();
		TotalLayout->setContentsMargins(0, 0, 0, 0);
		TotalLayout->setSpacing(5);
		TotalLayout->addLayout(Gridlayout);
		TotalLayout->addLayout(Hlayout1);
		TotalLayout->addLayout(Hlayout2);
		TotalLayout->addLayout(boolLayout);
		TotalLayout->addLayout(SpacingHlayout);

		this->setLayout(TotalLayout);
		
	}

	void QCurveWidget::updateField()
	{
		FVar<Ramp>* f = TypeInfo::cast<FVar<Ramp>>(field());
		if (f == nullptr)
		{
			return;
		}

	}

	void QCurveLabel::paintEvent(QPaintEvent* event)
	{

		//set Point Radius
		radius = 4;
		int w = this->width();
		int h = this->height();
		
		//set border
		minX = 0 + 1.5 * radius;
		maxX = w - 2 * radius;
		minY = 0 + 2 * radius;
		maxY = h - 1.5 * radius;

		//get data
		if (CoordArray.empty()) 
		{
			if (field->getValue().Originalcoord.empty())		//If there is no data from the Widget in the field, it is initialized from the field
			{

				this->copyFromField(field->getValue().MyCoord, reSortCoordArray);
				CoordArray.assign(reSortCoordArray.begin(), reSortCoordArray.end());
				buildCoordToResortMap();
				this->copyFromField(field->getValue().myHandlePoint, HandlePoints);
			}
			else		//use data from the Widget
			{
				this->copyFromField(field->getValue().Originalcoord,CoordArray);
				//this->copyFromField(field->getDataPtr()->OriginalHandlePoint, HandlePoints);
				this->copyFromField(field->getValue().OriginalHandlePoint, HandlePoints);
			}

		}

		//if useClose,add end & start Point.
		reSortCoordArray.assign(CoordArray.begin(), CoordArray.end());		
		reSort(reSortCoordArray);		


		QPainter painter(this);
		painter.setRenderHint(QPainter::Antialiasing, true);

		//Background
		QBrush brush = QBrush(Qt::black, Qt::SolidPattern);
		painter.setBrush(brush);
		QRectF Bound = QRectF(QPointF(minX, minY), QPointF(maxX, maxY));
		painter.drawRect(Bound);

		//Grid
		QBrush brush2 = QBrush(QColor(100,100,100), Qt::CrossPattern);
		painter.setBrush(brush2);
		painter.drawRect(Bound);

		//Draw Ellipse
		size_t ptNum = CoordArray.size();

		QVector<QPointF> QCoordArray;
		for (size_t i = 0; i < reSortCoordArray.size(); i++) 
		{
			QCoordArray.push_back(QPointF(reSortCoordArray[i].x, reSortCoordArray[i].y));
		}


		buildCoordToResortMap();	//Build map. find element id of sorted reSortCoordArray by CoordArray

		if (HandlePoints.empty())	//Build HandlePoints for Bezier handles from CoordArray.
		{
			buildHandlePointSet();
		}

		
		QPen LinePen = QPen(QPen(QBrush(QColor(200,200,200)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePen);

		path.clear();
		if (useBezier)		//Draw Bezier or Line
		{
			//draw Bezier
			for (size_t i = 1; i < reSortCoordArray.size(); i++)
			{
				int ptnum = i - 1;
				path.moveTo(reSortCoordArray[ptnum].x, reSortCoordArray[ptnum].y);

				auto it = mapResortIDtoOriginalID.find(i);
				int id = it->second;
				int s = id * 2;

				auto itf = mapResortIDtoOriginalID.find(ptnum);
				int idf = itf->second;
				int f = idf * 2 + 1;

				path.cubicTo(QPointF(HandlePoints[f].x, HandlePoints[f].y), QPointF(HandlePoints[s].x, HandlePoints[s].y), QPointF(reSortCoordArray[ptnum + 1].x, reSortCoordArray[ptnum + 1].y));
				painter.drawPath(path);
			}
			// use CurveClose?
			if (curveClose && reSortCoordArray.size()>=3)
			{
				int end = reSortCoordArray.size() - 1;
				int handleEnd = HandlePoints.size() - 1;
				path.moveTo(reSortCoordArray[end].x, reSortCoordArray[end].y);
				path.cubicTo(QPointF(HandlePoints[handleEnd].x, HandlePoints[handleEnd].y), QPointF(HandlePoints[0].x, HandlePoints[0].y), QPointF(reSortCoordArray[0].x, reSortCoordArray[0].y));
				painter.drawPath(path);
			}
		}		
		else 
		{
			if (QCoordArray.size() >= 2) 
			{
				for (size_t i = 1; i < QCoordArray.size(); i++)
				{
					//painter.drawLine(QCoordArray[i], QCoordArray[i + 1]);

					int ptnum = i - 1;
					path.moveTo(reSortCoordArray[ptnum].x, reSortCoordArray[ptnum].y);
					path.lineTo(QPointF(reSortCoordArray[ptnum + 1].x, reSortCoordArray[ptnum + 1].y));
					painter.drawPath(path);
				}

				if (curveClose && reSortCoordArray.size() >= 3)
				{
					int end = reSortCoordArray.size()-1;
					path.moveTo(reSortCoordArray[end].x, reSortCoordArray[end].y);
					path.lineTo(QPointF(reSortCoordArray[0].x, reSortCoordArray[0].y));
					painter.drawPath(path);
				}
			}
		}

		QPen LinePenWhite = QPen(QPen(QBrush(Qt::white), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePenWhite);

		//draw Point
		for (size_t i = 0; i < ptNum; i++)
		{
			painter.setBrush(QBrush(Qt::gray, Qt::SolidPattern));
			painter.drawEllipse(CoordArray[i].x - radius, CoordArray[i].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(QColor(200, 200, 200), Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[i].x - radius, CoordArray[i].y - radius, 2 * radius, 2 * radius);
		}

		//Paint SelectPoint
		if (selectPoint != -1)
		{
			painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
			painter.drawEllipse(CoordArray[selectPoint].x - radius, CoordArray[selectPoint].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(Qt::white, Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[selectPoint].x - radius, CoordArray[selectPoint].y - radius, 2 * radius, 2 * radius);
		}
		//Paint hoverPoint
		if (hoverPoint != -1)
		{
			painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
			painter.drawEllipse(CoordArray[hoverPoint].x - radius, CoordArray[hoverPoint].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(Qt::white, Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[hoverPoint].x - radius, CoordArray[hoverPoint].y - radius, 2 * radius, 2 * radius);
		}

		//drawHandle
		if (useBezier) 
		{
	

			if (handleParent != -1)
			{
				int f = handleParent * 2;
				int s = handleParent * 2 + 1;
				//draw Handle
				painter.drawLine(QPointF(CoordArray[handleParent].x, CoordArray[handleParent].y), QPointF(HandlePoints[f].x, HandlePoints[f].y));
				painter.drawLine(QPointF(CoordArray[handleParent].x, CoordArray[handleParent].y), QPointF(HandlePoints[s].x, HandlePoints[s].y));
				//draw ControlPoint
				painter.drawEllipse(HandlePoints[f].x - radius, HandlePoints[f].y - radius, 2 * radius, 2 * radius);
				painter.drawEllipse(HandlePoints[s].x - radius, HandlePoints[s].y - radius, 2 * radius, 2 * radius);
				//draw ParentPoint
				painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
				painter.drawEllipse(CoordArray[handleParent].x - radius, CoordArray[handleParent].y - radius, 2 * radius, 2 * radius);


				QPen LinePen2 = QPen(QPen(QBrush(QColor(255, 255, 255)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
				painter.setPen(LinePen2);
				painter.drawEllipse(CoordArray[handleParent].x - radius, CoordArray[handleParent].y - radius, 2 * radius, 2 * radius);
			}
			if (selectHandlePoint != -1 )
			{
				QPen LinePen2 = QPen(QPen(QBrush(QColor(255, 255, 255)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
				painter.setPen(LinePen2);
				painter.drawEllipse(HandlePoints[selectHandlePoint].x - radius, HandlePoints[selectHandlePoint].y - radius, 2 * radius, 2 * radius);

			}
		}

		if (ForceUpdate)
		{ 

			buildBezierPoint();			
			updateDataToField(); 
			ForceUpdate = false;
			
		}

		painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
		painter.setPen(QPen(QPen(QBrush(QColor(255, 255, 255)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin)));

		//draw selected point
		if (multiSelectID.size()) 
		{
			for (auto it : multiSelectID)
			{
				painter.drawEllipse(CoordArray[it].x - radius, CoordArray[it].y - radius, 2 * radius, 2 * radius);
			}
		}


		painter.setBrush(QBrush(QColor(255, 0, 0), Qt::SolidPattern));
		painter.setPen(QPen(QPen(QBrush(QColor(255, 0, 0)), 1, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin)));
		


	}

	QCurveLabel::~QCurveLabel()
	{

	}


	QCurveLabel::QCurveLabel(QWidget* parent)
	{
		this->setLabelSize(w, w, w, w);
		this->setStyleSheet("background:rgba(110,115,100,1)");
		this->setMouseTracking(true);

	}

	void QCurveLabel::setLabelSize(int minX, int minY, int maxX, int maxY)
	{
		this->setMinimumSize(minX, minY);
		this->setMaximumSize(maxX, maxY);
	}

	void QCurveLabel::reSort(std::vector<MyCoord>& vector1)
	{

		if (useSort) 
		{
			if (Mode == x)
			{
				std::sort(vector1.begin(), vector1.end(), sortx);
			}

			if (Mode == y)
			{
				sort(vector1.begin(), vector1.end(), sorty);
			}
		}

	}

	void QCurveLabel::mousePressEvent(QMouseEvent* event)
	{
		//MouseLeft
		pressCoord.x = event->pos().x();
		pressCoord.y = event->pos().y();

		if (shiftKey)	
		{
			for (size_t i = 0; i < CoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((pressCoord.x - CoordArray[i].x), 2) + std::pow((pressCoord.y - CoordArray[i].y), 2));
				if (temp < selectDistance)
				{
					selectPoint = i;
					isSelect = true;
					handleParent = selectPoint;
					iniPos.set(CoordArray[selectPoint]);

					bool elementInMulti = false;

					std::vector<int>::iterator iter;
					iter = multiSelectID.begin();
					while (iter != multiSelectID.end())
					{
						if (*iter == i)
						{
							elementInMulti = true;
							break;
						}
						iter++;
					}
					if (elementInMulti) 
					{

						multiSelectID.erase(iter);
						elementInMulti = false;
					}
					else
					{
						multiSelectID.push_back(i); 
					}
					break;
				}
			}
		}
		else if (!shiftKey && !altKey)		//
		{
			for (size_t i = 0; i < CoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((pressCoord.x - CoordArray[i].x), 2) + std::pow((pressCoord.y - CoordArray[i].y), 2));

				if (temp < selectDistance)
				{
					selectPoint = i;
					isSelect = true;
					handleParent = selectPoint;
					iniPos.set(CoordArray[selectPoint]);
					{
						if (std::find(multiSelectID.begin(), multiSelectID.end(), i) != multiSelectID.end())
						{
						}
						else
						{
							multiSelectID.clear();
							multiSelectID.push_back(i);
						}
					}

					if (selectPoint == 0)
					{
						InsertAtBegin = true;
					}
					else if (selectPoint == CoordArray.size() - 1)
					{
						InsertAtBegin = false;
					}

					break;
				}
			}
		}
		
		//if select Handle
		if (useBezier)
		{
			//printf("handle\n");
			int displayHandle[2] = { handleParent * 2 ,handleParent * 2 + 1};
			for (size_t k = 0; k < 2; k++)
			{
				int i = displayHandle[k];

				if (i < 0) { break; }
				
				int temp = sqrt(std::pow((pressCoord.x - HandlePoints[i].x), 2) + std::pow((pressCoord.y - HandlePoints[i].y), 2));

				if (temp < selectDistance && isSelect != true)
				{
					selectHandlePoint = i;
					isHandleSelect = true;
					iniHandlePos.set(HandlePoints[selectHandlePoint]);

					if (selectHandlePoint % 2 == 0) { handleParent = selectHandlePoint / 2; }
					else { handleParent = (selectHandlePoint - 1) / 2; }


					if (selectHandlePoint % 2 == 0)
					{
						connectHandlePoint = selectHandlePoint + 1;
					}
					else
					{
						connectHandlePoint = selectHandlePoint - 1;
					}

					Vec2f V2 = Vec2f(HandlePoints[connectHandlePoint].x - CoordArray[handleParent].x, HandlePoints[connectHandlePoint].y - CoordArray[handleParent].y);

					connectLength = V2.norm();
				
					{//handle Connect?
						Vec2f V3 = Vec2f(HandlePoints[selectHandlePoint].x - CoordArray[handleParent].x, HandlePoints[selectHandlePoint].y - CoordArray[handleParent].y);
						Vec2f V4 = Vec2f(HandlePoints[connectHandlePoint].x - CoordArray[handleParent].x, HandlePoints[connectHandlePoint].y - CoordArray[handleParent].y);
						V4.normalize();
						Vec2f N = -1 * V3.normalize();
						float angle = acos(V4.dot(N)) / M_PI * 180;
						if (angle <= HandleAngleThreshold) { HandleConnection = true; }
						else { HandleConnection = false; }
					}

					break;
				}
			}
		}
		if (!isSelect && !isHandleSelect)
		{

			//pushback Point
			if (!ctrlKey)
			{
				addPointtoEnd();
			}
			//insert Point in Edge
			else if (ctrlKey)
			{
				if (reSortCoordArray.size() >= 2)
				{
					int id = insertCurvePoint(pressCoord);

					if (InterpMode == Interpolation::Bezier)
					{
						selectPoint = -1;
						isSelect = false;
						InsertBezierOpenPoint = true;
						handleParent = id;
						selectHandlePoint = id * 2 + 1;
						connectHandlePoint = selectHandlePoint - 1;
						isHandleSelect = true;
					}
					if (InterpMode == Interpolation::Linear)
					{
						selectPoint = id;
						isSelect = true;
						InsertBezierOpenPoint = false;
						handleParent = id;
						selectHandlePoint = -1;
						connectHandlePoint = -1;
						isHandleSelect = false;
					}

					multiSelectID.clear();
					multiSelectID.push_back(id);
				}
				else
				{ 
					addPointtoEnd(); 
				}
				
			}

		}
		
		else if(event->button() == Qt::RightButton)
		{

			if(selectPoint >=0)
			{
				deletePoint();
			}

		}


		this->update();
	}
	
	int QCurveLabel::addPointtoEnd()
	{	
		if (!InsertAtBegin)
		{
			CoordArray.push_back(pressCoord);
			buildCoordToResortMap();
			insertHandlePoint(CoordArray.size() - 1,pressCoord);

			if (InterpMode == Interpolation::Bezier)
			{
				InsertBezierOpenPoint = true;
				selectPoint = -1;
				isSelect = false;
				handleParent = CoordArray.size() - 1;
				selectHandlePoint = handleParent * 2 + 1;
				connectHandlePoint = selectHandlePoint - 1;
				isHandleSelect = true;
			}
			if (InterpMode == Interpolation::Linear)
			{
				InsertBezierOpenPoint = false;
				selectPoint = CoordArray.size() - 1;
				isSelect = true;
				handleParent = selectPoint;
				selectHandlePoint = -1;
				connectHandlePoint = -1;
				isHandleSelect = false;
				iniPos.set(CoordArray[selectPoint]);
			}

			multiSelectID.clear();
			multiSelectID.push_back(CoordArray.size() - 1);
		}
		else if(InsertAtBegin)
		{
			CoordArray.insert(CoordArray.begin(), pressCoord);
			buildCoordToResortMap();
			insertHandlePoint(0, pressCoord);


			if (InterpMode == Interpolation::Bezier)
			{
				InsertBezierOpenPoint = true;
				selectPoint = -1;
				isSelect = false;
				handleParent = 0;
				selectHandlePoint = 1;
				connectHandlePoint = selectHandlePoint - 1;
				isHandleSelect = true;
			}
			if (InterpMode == Interpolation::Linear)
			{
				InsertBezierOpenPoint = false;
				selectPoint = 0;
				isSelect = true;
				handleParent = selectPoint;
				selectHandlePoint = -1;
				connectHandlePoint = -1;
				isHandleSelect = false;
				iniPos.set(CoordArray[selectPoint]);
			}
		}

		
		return handleParent;

	}

	void QCurveLabel::deletePoint()
	{
		if (multiSelectID.size() <= 1) 
		{
			CoordArray.erase(CoordArray.begin() + selectPoint);
			HandlePoints.erase(HandlePoints.begin() + selectPoint * 2 + 1);
			HandlePoints.erase(HandlePoints.begin() + selectPoint * 2);
			reSortCoordArray.clear();
			reSortCoordArray.assign(CoordArray.begin(), CoordArray.end());
			reSort(reSortCoordArray);
			buildCoordToResortMap();
		}
		else 
		{
			std::sort(multiSelectID.begin(), multiSelectID.end());

			for (size_t i = 0; i < multiSelectID.size(); i++) 
			{
				selectPoint = multiSelectID[multiSelectID.size() - i - 1];
				CoordArray.erase(CoordArray.begin() + selectPoint);
				HandlePoints.erase(HandlePoints.begin() + selectPoint * 2 + 1);
				HandlePoints.erase(HandlePoints.begin() + selectPoint * 2);
				reSortCoordArray.clear();
				reSortCoordArray.assign(CoordArray.begin(), CoordArray.end());
				reSort(reSortCoordArray);
				buildCoordToResortMap();
			}

		}
		

		multiSelectID.clear();

		selectPoint = -1;
		hoverPoint = -1;
		isHover = false;
		isSelect = false;
		handleParent = -1;
		selectHandlePoint = -1;
		connectHandlePoint = -1;
		isHandleSelect = false;

	}

	void QCurveLabel::mouseMoveEvent(QMouseEvent* event)
	{
		this->grabKeyboard();
		//constrained
		if (isSelect) 
		{
			CoordArray[selectPoint].x = dyno::clamp(event->pos().x(), minX, maxX);
			CoordArray[selectPoint].y = dyno::clamp(event->pos().y(), minY, maxY);

			dtPos.set(CoordArray[selectPoint].x-iniPos.x , CoordArray[selectPoint].y - iniPos.y);

			HandlePoints[selectPoint * 2].x = dyno::clamp(HandlePoints[selectPoint * 2].x + dtPos.x, minX, maxX);
			HandlePoints[selectPoint * 2].y = dyno::clamp(HandlePoints[selectPoint * 2].y + dtPos.y, minY, maxY);

			HandlePoints[selectPoint * 2 + 1].x = dyno::clamp(HandlePoints[selectPoint * 2 + 1].x + dtPos.x, minX, maxX);
			HandlePoints[selectPoint * 2 + 1].y = dyno::clamp(HandlePoints[selectPoint * 2 + 1].y + dtPos.y, minY, maxY);

			iniPos = CoordArray[selectPoint];

			if (multiSelectID.size() > 1)
			{
				for (size_t i = 0; i < multiSelectID.size(); i++)
				{
					int tempSelect = multiSelectID[i];
					if (tempSelect != selectPoint) 
					{
			

						CoordArray[tempSelect].x = dyno::clamp(CoordArray[tempSelect].x + dtPos.x, minX, maxX);
						CoordArray[tempSelect].y = dyno::clamp(CoordArray[tempSelect].y + dtPos.y, minY, maxY);

						//
						HandlePoints[tempSelect * 2].x = dyno::clamp(HandlePoints[tempSelect * 2].x + dtPos.x, minX, maxX);
						HandlePoints[tempSelect * 2].y = dyno::clamp(HandlePoints[tempSelect * 2].y + dtPos.y, minY, maxY);

						HandlePoints[tempSelect * 2 + 1].x = dyno::clamp(HandlePoints[tempSelect * 2 + 1].x + dtPos.x, minX, maxX);
						HandlePoints[tempSelect * 2 + 1].y = dyno::clamp(HandlePoints[tempSelect * 2 + 1].y + dtPos.y, minY, maxY);
					}
				}

			}





			update();
		}
		//constrained Handle
		if (isHandleSelect)
		{

			HandlePoints[selectHandlePoint].x = dyno::clamp(event->pos().x(), minX, maxX); 
			HandlePoints[selectHandlePoint].y = dyno::clamp(event->pos().y(), minY, maxY);
			
			Vec2f V1 = Vec2f(HandlePoints[selectHandlePoint].x - CoordArray[handleParent].x, HandlePoints[selectHandlePoint].y - CoordArray[handleParent].y);
			Vec2f V2 = V1;
			Vec2f V3 = Vec2f(HandlePoints[connectHandlePoint].x - CoordArray[handleParent].x, HandlePoints[connectHandlePoint].y - CoordArray[handleParent].y);
			V3.normalize();
			Vec2f P = Vec2f(CoordArray[handleParent].x, CoordArray[handleParent].y);
			Vec2f N = -1 * V1.normalize();
			float angle = acos(V3.dot(N))/ M_PI *180;

			if (!altKey && HandleConnection) //!altKey
			{
				if (InsertBezierOpenPoint)
				{
					HandlePoints[connectHandlePoint] = N * (V2.norm()) + P;
				}
				else 
				{
					HandlePoints[connectHandlePoint] = N * connectLength + P;
					HandlePoints[connectHandlePoint].x = dyno::clamp(HandlePoints[connectHandlePoint].x, minX, maxX);
					HandlePoints[connectHandlePoint].y = dyno::clamp(HandlePoints[connectHandlePoint].y, minY, maxY);
				}
			}


			update();
		}


		if (isHover == true)
		{
			int tempHover = sqrt(std::pow((event->pos().x() - CoordArray[hoverPoint].x), 2) + std::pow((event->pos().y() - CoordArray[hoverPoint].y), 2));
			if (tempHover >= selectDistance)
			{
				hoverPoint = -1;
				isHover = false;
			}
		}
		else 
		{
			for (size_t i = 0; i < CoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((event->pos().x() - CoordArray[i].x), 2) + std::pow((event->pos().y() - CoordArray[i].y), 2));

				if (temp < selectDistance)
				{
					hoverPoint = i;
					isHover = true;
					break;
				}
			}
			update();
		}

	}
	void QCurveLabel::mouseReleaseEvent(QMouseEvent* event)
	{
		selectPoint = -1;
		isSelect = false;
		isHandleSelect = false;
		InsertBezierOpenPoint = false;
		selectHandlePoint = -1;
		connectHandlePoint = -1;
		HandleConnection = true;

		//build Bezier Point
		buildBezierPoint();
		updateDataToField();
		update();
	}
	void QCurveLabel::updateDataToField()
	{
		Curve s;

		if (field == nullptr) { return; }
		else
		{
			if (!reSortCoordArray.empty()) { updateFloatCoordArray(reSortCoordArray, floatCoord); }
			if (!HandlePoints.empty()) { updateFloatCoordArray(HandlePoints, handleFloatCoord); }

			 CoordtoField(s); 
		}



		field->setValue(s);

	}

	void QCurveLabel::changeValue(int s)
	{
		this->Mode = (Dir)s;
		Curve ras;

		initializeLine(Mode);

		if (field == nullptr) { return; }
		else
		{
			updateDataToField();
			if (!floatCoord.empty()) { CoordtoField(ras); }
		}

		this->ForceUpdate = true;
		update();

	}

	void QCurveLabel::changeInterpValue(int s)
	{
		this->InterpMode = (Interpolation)s;
		if (InterpMode == Interpolation::Linear) { useBezier = 0; }
		else if (InterpMode == Interpolation::Bezier) { useBezier = 1; }
		this->ForceUpdate = true;
		update();
	}

	void QCurveLabel::initializeLine(Dir mode)
	{
		if (mode == x)
		{
			CoordArray[0].x = minX;
			CoordArray[0].y = (maxY + minY) / 2 ;
			CoordArray[1].x = maxX;
			CoordArray[1].y = (maxY + minY) / 2;
		}
		if (mode == y)
		{
			CoordArray[0].x = (maxX + minX) / 2 ;
			CoordArray[0].y = minY;
			CoordArray[1].x = (maxX + minX) / 2;
			CoordArray[1].y = maxY;
		}
	}

	void QCurveLabel::changeLabelSize()
	{
		if (!lockSize) 
		{
			isSquard = !isSquard;

			updateLabelShape();
		}	
	}
	void QCurveLabel::updateLabelShape()
	{
		if (isSquard)
		{
			this->setLabelSize(w, w, w, w);
			remapArrayToHeight(CoordArray,w);
			remapArrayToHeight(reSortCoordArray, w);
			remapArrayToHeight(HandlePoints, w);
		}
		else
		{
			this->setLabelSize(w, h, w, w);
			remapArrayToHeight(CoordArray, h);
			remapArrayToHeight(reSortCoordArray, h);
			remapArrayToHeight(HandlePoints, h);
		}
		
		//ForceUpdate = true;
		this->update();
	}

	void QCurveLabel::remapArrayToHeight(std::vector<MyCoord>& Array, int h)
	{
		double fmaxX = double(maxX);
		double fminX = double(minX);
		double fmaxY = double(maxY);
		double fminY = double(minY);
		for (size_t i = 0; i < Array.size(); i++)
		{
			int newMaxY = h - 1.5 * double(radius);
			float k = (double(Array[i].y) - fminY) / (fmaxY - fminY);
			Array[i].y = k * (newMaxY - fminY) + fminY;
		}
	}
	void QCurveLabel::buildBezierPoint()
	{
		int totalLength = path.length();
		CurvePoint.clear();
		curvePointMapLength.clear();
		for (size_t i = 0; i < 500; i++)
		{
			float length = i * spacing ;
			qreal perc = 0;
			QPointF QP;
			bool b = false;
			if (length <= totalLength)
			{
				perc = path.percentAtLength(qreal(length));
			}
			else
			{
				perc = 1;
				b = true;
			}

			QP = path.pointAtPercent(perc);
			CurvePoint.push_back(MyCoord(QP.x(), QP.y()));
			curvePointMapLength[i] = length;

			if (b) { break; }
		}
		
	}

	void QCurveLabel::setCurveClose()
	{
		curveClose = !curveClose;
		this->ForceUpdate = true;
		update();
	}


	void  QCurveLabel::keyReleaseEvent(QKeyEvent* event)
	{
		QWidget::keyPressEvent(event);
		parent()->event((QEvent*)event);
		if (event->key() == Qt::Key_Alt)
		{
			altKey = false;
			return;
		}
		if (event->key() == Qt::Key_Control)
		{
			ctrlKey = false;
			return;
		}
		if (event->key() == Qt::Key_Shift)
		{
			shiftKey = false;
			return;
		}
	}

	void  QCurveLabel::leaveEvent(QEvent* event)
	{
		shiftKey = false;
		altKey = false;
		this->releaseKeyboard();
	}


	void QCurveLabel::insertHandlePoint(int fp, MyCoord pCoord)
	{
		dyno::Vec2f P(pCoord.x, pCoord.y);
		int size = reSortCoordArray.size();
		dyno::Vec2f p1;
		dyno::Vec2f p2;

		dyno::Vec2f N = Vec2f(1, 0);
		if (fp == 0)
		{
			if (reSortCoordArray.size() == 1)
				N = Vec2f(1, 0);
			else
				N = Vec2f(pCoord.x - reSortCoordArray[1].x, pCoord.y - reSortCoordArray[1].y);
		}
		else
		{
			N = Vec2f(pCoord.x - reSortCoordArray[fp - 1].x, pCoord.y - reSortCoordArray[fp - 1].y) * -1;
		}


		N.normalize();

		int num = fp * 2;

		p1 = P - N * iniHandleLength;
		p2 = P + N * iniHandleLength;

		HandlePoints.insert(HandlePoints.begin() + num, MyCoord(p1));
		HandlePoints.insert(HandlePoints.begin() + num, MyCoord(p2));
	}

	int QCurveLabel::insertCurvePoint(MyCoord pCoord)
	{
		lengthMapEndPoint.clear();
		QPainterPath tempPath;
		for (size_t i = 1; i < reSortCoordArray.size(); i++)
		{

			int ptnum = i - 1;
			tempPath.moveTo(reSortCoordArray[ptnum].x, reSortCoordArray[ptnum].y);

			auto it = mapResortIDtoOriginalID.find(i);
			int id = it->second;
			int s = id * 2;

			auto itf = mapResortIDtoOriginalID.find(ptnum);
			int idf = itf->second;
			int f = idf * 2 + 1;
			if (InterpMode == Interpolation::Bezier)
			{
				tempPath.cubicTo(QPointF(HandlePoints[f].x, HandlePoints[f].y), QPointF(HandlePoints[s].x, HandlePoints[s].y), QPointF(reSortCoordArray[ptnum + 1].x, reSortCoordArray[ptnum + 1].y));
			}
			else if (InterpMode == Interpolation::Linear)
			{
				tempPath.lineTo(QPointF(reSortCoordArray[ptnum + 1].x, reSortCoordArray[ptnum + 1].y));
			}
			float tempLength = tempPath.length();

			EndPoint tempEP = EndPoint(id, idf);
			lengthMapEndPoint[tempLength] = tempEP;
		}

		int dis = 380;
		int nearPoint = -1;
		int temp;

		for (size_t i = 0; i < CurvePoint.size(); i++)
		{
			temp = sqrt(std::pow((pCoord.x - CurvePoint[i].x), 2) + std::pow((pCoord.y - CurvePoint[i].y), 2));

			if (dis >= temp)
			{
				nearPoint = i;
				dis = temp;
			}
		}

		int fp = -1;
		int searchRadius = 10;

		if (isSquard) { searchRadius = 20; }
		else { searchRadius = 10; }

		if (dis < searchRadius)
		{
			float realLength = curvePointMapLength.find(nearPoint)->second;
			float finalLength = -1;
			for (auto it : lengthMapEndPoint)
			{
				if (realLength <= it.first)
				{
					finalLength = it.first;
					break;
				}
			}
			if (finalLength == -1)
			{
				fp = addPointtoEnd();
			}
			else
			{
				fp = lengthMapEndPoint.find(finalLength)->second.firstPoint;

				CoordArray.insert(CoordArray.begin() + fp, pCoord);
				reSortCoordArray.clear();
				reSortCoordArray.assign(CoordArray.begin(), CoordArray.end());
				reSort(reSortCoordArray);
				buildCoordToResortMap();
				insertHandlePoint(fp, pCoord);

				iniPos.set(pCoord);
			}
		}
		else
		{
			fp = addPointtoEnd();
		}
		return fp;

	}

	void QCurveLabel::keyPressEvent(QKeyEvent* event)
	{
		QWidget::keyPressEvent(event);
		parent()->event((QEvent*)event);
		if (event->key() == Qt::Key_Alt)
		{
			altKey = true;
			return;
		}
		if (event->key() == Qt::Key_Control)
		{

			ctrlKey = true;
			return;
		}
		if (event->key() == Qt::Key_Shift)
		{
			shiftKey = true;
			return;
		}
	}

	void  QCurveLabel::updateFloatCoordArray(std::vector<MyCoord> CoordArray, std::vector<Coord0_1>& myfloatCoord)
	{
		myfloatCoord.clear();
		for (auto it : CoordArray)
		{
			myfloatCoord.push_back(CoordTo0_1Value(it));
		}
	}

	void QCurveLabel::CoordtoField(Curve& s)
	{
		s.clearMyCoord();
		for (auto it : floatCoord)
		{
			s.addFloatItemToCoord(it.x, it.y, s.MyCoord);
		}

		for (auto it : handleFloatCoord)
		{
			s.addFloatItemToCoord(it.x, it.y, s.myHandlePoint);
		}

		for (auto it : CoordArray)
		{
			s.addItemOriginalCoord(it.x, it.y);
		}
		for (auto it : HandlePoints)
		{
			s.addItemHandlePoint(it.x, it.y);
		}

		s.useCurve = useBezier;

		s.resample = LineResample;
		s.useSquard = isSquard;
		s.Spacing = spacing;

		s.NminX = NminX;
		s.NmaxX = NmaxX;
		s.NminY = NminY;
		s.NmaxY = NmaxY;

		s.curveClose = curveClose;

		if (useBezier) { s.InterpMode = Curve::Interpolation::Bezier; }
		else { s.InterpMode = Curve::Interpolation::Linear; }

		s.updateBezierCurve();
		s.UpdateFieldFinalCoord();
	}


	QCurveLabel::Coord0_1 QCurveLabel::CoordTo0_1Value(MyCoord& coord)
	{
		Coord0_1 s;

		double x = double(coord.x);
		double y = double(coord.y);
		double fmaxX = double(maxX);
		double fminX = double(minX);
		double fmaxY = double(maxY);
		double fminY = double(minY);

		s.x = (x - fminX) / (fmaxX - fminX) ;
		s.y = 1 - (y - fminY)  / (fmaxY - fminY);

		//s.x = (x - fminX) * (NmaxX - NminX) / (fmaxX - fminX) + NminX;
		//s.y = NmaxY - ((y - fminY) * (NmaxY - NminY) / (fmaxY - fminY) + NminY);

		return s;
	}

	QCurveLabel::MyCoord QCurveLabel::ZeroOneToCoord(Coord0_1& value, int x0, int x1, int y0, int y1)
	{

		MyCoord s;
		s.x = int(value.x * float(x1 - x0)) + x0;
		s.y = int((1 - value.y) * float(y1 - y0)) + y0;

		return s;
	}

	void QCurveLabel::buildCoordToResortMap()
	{
		reSortCoordArray.assign(CoordArray.begin(), CoordArray.end());
		reSort(reSortCoordArray);

		for (size_t i = 0; i < reSortCoordArray.size(); i++)
		{
			for (size_t k = 0; k < CoordArray.size(); k++)
			{
				if (reSortCoordArray[i] == CoordArray[k])
				{
					mapOriginalIDtoResortID[k] = i;
					mapResortIDtoOriginalID[i] = k;
					break;
				}
			}
		}

	};

	void QCurveLabel::insertElementToHandlePointSet(int i)
	{
		dyno::Vec2f P(CoordArray[i].x, CoordArray[i].y);

		dyno::Vec2f p1;
		dyno::Vec2f p2;
		dyno::Vec2f N;

		auto it = mapOriginalIDtoResortID.find(i);
		int id = it->second;
		int f;
		int s;
		if (CoordArray.size() < 3)
		{
			N = Vec2f(1, 0);
		}
		else
		{
			f = id - 1;
			s = id + 1;
			if (id == 0)
			{
				N[0] = reSortCoordArray[s].x - reSortCoordArray[id].x;
				N[1] = reSortCoordArray[s].y - reSortCoordArray[id].y;
			}
			else if (id == reSortCoordArray.size() - 1)
			{
				N[0] = reSortCoordArray[id].x - reSortCoordArray[f].x;
				N[1] = reSortCoordArray[id].y - reSortCoordArray[f].y;
			}
			else
			{
				N[0] = reSortCoordArray[s].x - reSortCoordArray[f].x;
				N[1] = reSortCoordArray[s].y - reSortCoordArray[f].y;
			}
		}

		N.normalize();

		p1 = P - N * iniHandleLength;
		p2 = P + N * iniHandleLength;

		HandlePoints.push_back(MyCoord(p1));
		HandlePoints.push_back(MyCoord(p2));

	}

	void QCurveLabel::setSpacingToDrawLabel(double value, int id)
	{
		spacing = value;
		ForceUpdate = true;
		update();
	}

	void QCurveLabel::SetValueToDrawLabel(double value, int id)
	{
		switch (id)
		{
		case 0: NminX = value;
			break;
		case 1: NminY = value;
			break;
		case 2: NmaxX = value;
			break;
		case 3: NmaxY = value;
			break;
		}
		ForceUpdate = true;
		update();
	}

	void QCurveLabel::copyFromField(std::vector<Curve::Coord2D> coord01, std::vector<MyCoord>& thisArray)
	{	
		if (coord01.size())
		{
			for (auto it : coord01)
			{
				Coord0_1 s;
				s.set(it.x, it.y);
				thisArray.push_back(ZeroOneToCoord(s, minX, maxX, minY, maxY));
			}
		}

	}

	void QCurveLabel::copyFromField(std::vector<Curve::OriginalCoord> coord01, std::vector<MyCoord>& thisArray)
	{
		if (coord01.size())
		{
			for (auto it : coord01)
			{
				MyCoord s;
				s.set(it.x, it.y);
				thisArray.push_back(s);
			}

		}
	}

	void QCurveLabel::buildHandlePointSet()
	{
		for (size_t i = 0; i < CoordArray.size(); i++)
		{
			insertElementToHandlePointSet(i);
		}
	}


	void QCurveLabel::setLinearResample(int s)
	{
		LineResample = s;
		this->ForceUpdate = true;
		update();
	}

	void QCurveLabel::copySettingFromField()
	{
		useBezier = field->getValue().useCurve;
		LineResample = field->getValue().resample;
		spacing = field->getValue().Spacing;
		curveClose = field->getValue().curveClose;
		isSquard = field->getValue().useSquard;
		lockSize = field->getValue().lockSize;

		if (field->getValue().InterpMode == Ramp::Interpolation::Bezier) { InterpMode = Bezier; }
		else { InterpMode = Linear; }


		useSort = false;

	}

}

