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
		int curIndex2 = int(f->getValue().mInterpMode);
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
					sliderReal[i]->setValue(f->getValue().mNewMinY);
					spinnerReal[i]->setValue(f->getValue().mNewMinY);
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

	QCurveLabel::QCurveLabel(FVar<Curve>* f, QWidget* parent)
	{
		this->setField(f);
		this->copySettingFromField();
		if (isSquard)
			this->setLabelSize(w, w, w, w);
		else
			this->setLabelSize(w, h, w, w);
		this->setStyleSheet("background:rgba(110,115,100,1)");
		this->setMouseTracking(true);
	};

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
		if (mCoordArray.empty())
		{
			if (mField->getValue().Originalcoord.empty())		//If there is no data from the Widget in the field, it is initialized from the field
			{

				this->copyFromField(mField->getValue().mCoord, mReSortCoordArray);
				mCoordArray.assign(mReSortCoordArray.begin(), mReSortCoordArray.end());
				buildCoordToResortMap();
				this->copyFromField(mField->getValue().myHandlePoint, mHandlePoints);
			}
			else		//use data from the Widget
			{
				this->copyFromField(mField->getValue().Originalcoord, mCoordArray);
				//this->copyFromField(field->getDataPtr()->OriginalHandlePoint, HandlePoints);
				this->copyFromField(mField->getValue().OriginalHandlePoint, mHandlePoints);
			}

		}

		//if useClose,add end & start Point.
		mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());
		reSort(mReSortCoordArray);


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
		size_t ptNum = mCoordArray.size();

		QVector<QPointF> QCoordArray;
		for (size_t i = 0; i < mReSortCoordArray.size(); i++)
		{
			QCoordArray.push_back(QPointF(mReSortCoordArray[i].x, mReSortCoordArray[i].y));
		}


		buildCoordToResortMap();	//Build map. find element id of sorted reSortCoordArray by CoordArray

		if (mHandlePoints.empty())	//Build HandlePoints for Bezier handles from CoordArray.
		{
			buildHandlePointSet();
		}

		
		QPen LinePen = QPen(QPen(QBrush(QColor(200,200,200)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePen);

		mPath.clear();
		if (useBezier)		//Draw Bezier or Line
		{
			//draw Bezier
			for (size_t i = 1; i < mReSortCoordArray.size(); i++)
			{
				int ptnum = i - 1;
				mPath.moveTo(mReSortCoordArray[ptnum].x, mReSortCoordArray[ptnum].y);

				auto it = mMapResortIDtoOriginalID.find(i);
				int id = it->second;
				int s = id * 2;

				auto itf = mMapResortIDtoOriginalID.find(ptnum);
				int idf = itf->second;
				int f = idf * 2 + 1;

				mPath.cubicTo(QPointF(mHandlePoints[f].x, mHandlePoints[f].y), QPointF(mHandlePoints[s].x, mHandlePoints[s].y), QPointF(mReSortCoordArray[ptnum + 1].x, mReSortCoordArray[ptnum + 1].y));
				painter.drawPath(mPath);
			}
			// use CurveClose?
			if (curveClose && mReSortCoordArray.size()>=3)
			{
				int end = mReSortCoordArray.size() - 1;
				int handleEnd = mHandlePoints.size() - 1;
				mPath.moveTo(mReSortCoordArray[end].x, mReSortCoordArray[end].y);
				mPath.cubicTo(QPointF(mHandlePoints[handleEnd].x, mHandlePoints[handleEnd].y), QPointF(mHandlePoints[0].x, mHandlePoints[0].y), QPointF(mReSortCoordArray[0].x, mReSortCoordArray[0].y));
				painter.drawPath(mPath);
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
					mPath.moveTo(mReSortCoordArray[ptnum].x, mReSortCoordArray[ptnum].y);
					mPath.lineTo(QPointF(mReSortCoordArray[ptnum + 1].x, mReSortCoordArray[ptnum + 1].y));
					painter.drawPath(mPath);
				}

				if (curveClose && mReSortCoordArray.size() >= 3)
				{
					int end = mReSortCoordArray.size()-1;
					mPath.moveTo(mReSortCoordArray[end].x, mReSortCoordArray[end].y);
					mPath.lineTo(QPointF(mReSortCoordArray[0].x, mReSortCoordArray[0].y));
					painter.drawPath(mPath);
				}
			}
		}

		QPen LinePenWhite = QPen(QPen(QBrush(Qt::white), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePenWhite);

		//draw Point
		for (size_t i = 0; i < ptNum; i++)
		{
			painter.setBrush(QBrush(Qt::gray, Qt::SolidPattern));
			painter.drawEllipse(mCoordArray[i].x - radius, mCoordArray[i].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(QColor(200, 200, 200), Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(mCoordArray[i].x - radius, mCoordArray[i].y - radius, 2 * radius, 2 * radius);
		}

		//Paint SelectPoint
		if (mSelectPointID != -1)
		{
			painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
			painter.drawEllipse(mCoordArray[mSelectPointID].x - radius, mCoordArray[mSelectPointID].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(Qt::white, Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(mCoordArray[mSelectPointID].x - radius, mCoordArray[mSelectPointID].y - radius, 2 * radius, 2 * radius);
		}
		//Paint hoverPoint
		if (mHoverPoint != -1)
		{
			painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
			painter.drawEllipse(mCoordArray[mHoverPoint].x - radius, mCoordArray[mHoverPoint].y - radius, 2 * radius, 2 * radius);
			painter.setPen(QPen(QBrush(Qt::white, Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(mCoordArray[mHoverPoint].x - radius, mCoordArray[mHoverPoint].y - radius, 2 * radius, 2 * radius);
		}

		//drawHandle
		if (useBezier) 
		{
	

			if (mHandleParent != -1)
			{
				int f = mHandleParent * 2;
				int s = mHandleParent * 2 + 1;
				//draw Handle
				painter.drawLine(QPointF(mCoordArray[mHandleParent].x, mCoordArray[mHandleParent].y), QPointF(mHandlePoints[f].x, mHandlePoints[f].y));
				painter.drawLine(QPointF(mCoordArray[mHandleParent].x, mCoordArray[mHandleParent].y), QPointF(mHandlePoints[s].x, mHandlePoints[s].y));
				//draw ControlPoint
				painter.drawEllipse(mHandlePoints[f].x - radius, mHandlePoints[f].y - radius, 2 * radius, 2 * radius);
				painter.drawEllipse(mHandlePoints[s].x - radius, mHandlePoints[s].y - radius, 2 * radius, 2 * radius);
				//draw ParentPoint
				painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
				painter.drawEllipse(mCoordArray[mHandleParent].x - radius, mCoordArray[mHandleParent].y - radius, 2 * radius, 2 * radius);


				QPen LinePen2 = QPen(QPen(QBrush(QColor(255, 255, 255)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
				painter.setPen(LinePen2);
				painter.drawEllipse(mCoordArray[mHandleParent].x - radius, mCoordArray[mHandleParent].y - radius, 2 * radius, 2 * radius);
			}
			if (mSelectHandlePoint != -1 )
			{
				QPen LinePen2 = QPen(QPen(QBrush(QColor(255, 255, 255)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
				painter.setPen(LinePen2);
				painter.drawEllipse(mHandlePoints[mSelectHandlePoint].x - radius, mHandlePoints[mSelectHandlePoint].y - radius, 2 * radius, 2 * radius);

			}
		}

		if (mForceUpdate)
		{ 

			buildBezierPoint();			
			updateDataToField(); 
			mForceUpdate = false;
			
		}

		painter.setBrush(QBrush(QColor(80, 179, 255), Qt::SolidPattern));
		painter.setPen(QPen(QPen(QBrush(QColor(255, 255, 255)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin)));

		//draw selected point
		if (mMultiSelectID.size())
		{
			for (auto it : mMultiSelectID)
			{
				painter.drawEllipse(mCoordArray[it].x - radius, mCoordArray[it].y - radius, 2 * radius, 2 * radius);
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


	void QCurveLabel::mousePressEvent(QMouseEvent* event)
	{
		//MouseLeft
		mPressCoord.x = event->pos().x();
		mPressCoord.y = event->pos().y();

		if (mShiftKey)	
		{
			for (size_t i = 0; i < mCoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((mPressCoord.x - mCoordArray[i].x), 2) + std::pow((mPressCoord.y - mCoordArray[i].y), 2));
				if (temp < selectDistance)
				{
					mSelectPointID = i;
					mIsSelect = true;
					mHandleParent = mSelectPointID;
					mInitPosition.set(mCoordArray[mSelectPointID]);

					bool elementInMulti = false;

					std::vector<int>::iterator iter;
					iter = mMultiSelectID.begin();
					while (iter != mMultiSelectID.end())
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

						mMultiSelectID.erase(iter);
						elementInMulti = false;
					}
					else
					{
						mMultiSelectID.push_back(i);
					}
					break;
				}
			}
		}
		else if (!mShiftKey && !mAltKey)		//
		{
			for (size_t i = 0; i < mCoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((mPressCoord.x - mCoordArray[i].x), 2) + std::pow((mPressCoord.y - mCoordArray[i].y), 2));

				if (temp < selectDistance)
				{
					mSelectPointID = i;
					mIsSelect = true;
					mHandleParent = mSelectPointID;
					mInitPosition.set(mCoordArray[mSelectPointID]);
					{
						if (std::find(mMultiSelectID.begin(), mMultiSelectID.end(), i) != mMultiSelectID.end())
						{
						}
						else
						{
							mMultiSelectID.clear();
							mMultiSelectID.push_back(i);
						}
					}

					if (mSelectPointID == 0)
					{
						mInsertAtBegin = true;
					}
					else if (mSelectPointID == mCoordArray.size() - 1)
					{
						mInsertAtBegin = false;
					}

					break;
				}
			}
		}
		
		//if select Handle
		if (useBezier)
		{
			//printf("handle\n");
			int displayHandle[2] = { mHandleParent * 2 ,mHandleParent * 2 + 1};
			for (size_t k = 0; k < 2; k++)
			{
				int i = displayHandle[k];

				if (i < 0) { break; }
				
				int temp = sqrt(std::pow((mPressCoord.x - mHandlePoints[i].x), 2) + std::pow((mPressCoord.y - mHandlePoints[i].y), 2));

				if (temp < selectDistance && mIsSelect != true)
				{
					mSelectHandlePoint = i;
					mIsHandleSelect = true;
					mInitHandlePos.set(mHandlePoints[mSelectHandlePoint]);

					if (mSelectHandlePoint % 2 == 0) { mHandleParent = mSelectHandlePoint / 2; }
					else { mHandleParent = (mSelectHandlePoint - 1) / 2; }


					if (mSelectHandlePoint % 2 == 0)
					{
						mConnectHandlePoint = mSelectHandlePoint + 1;
					}
					else
					{
						mConnectHandlePoint = mSelectHandlePoint - 1;
					}

					Vec2f V2 = Vec2f(mHandlePoints[mConnectHandlePoint].x - mCoordArray[mHandleParent].x, mHandlePoints[mConnectHandlePoint].y - mCoordArray[mHandleParent].y);

					mConnectLength = V2.norm();
				
					{//handle Connect?
						Vec2f V3 = Vec2f(mHandlePoints[mSelectHandlePoint].x - mCoordArray[mHandleParent].x, mHandlePoints[mSelectHandlePoint].y - mCoordArray[mHandleParent].y);
						Vec2f V4 = Vec2f(mHandlePoints[mConnectHandlePoint].x - mCoordArray[mHandleParent].x, mHandlePoints[mConnectHandlePoint].y - mCoordArray[mHandleParent].y);
						V4.normalize();
						Vec2f N = -1 * V3.normalize();
						float angle = acos(V4.dot(N)) / M_PI * 180;
						if (angle <= HandleAngleThreshold) { mHandleConnection = true; }
						else { mHandleConnection = false; }
					}

					break;
				}
			}
		}
		if (!mIsSelect && !mIsHandleSelect)
		{

			//pushback Point
			if (!mCtrlKey)
			{
				addPointtoEnd();
			}
			//insert Point in Edge
			else if (mCtrlKey)
			{
				if (mReSortCoordArray.size() >= 2)
				{
					int id = insertCurvePoint(mPressCoord);

					if (InterpMode == Interpolation::Bezier)
					{
						mSelectPointID = -1;
						mIsSelect = false;
						mInsertBezierOpenPoint = true;
						mHandleParent = id;
						mSelectHandlePoint = id * 2 + 1;
						mConnectHandlePoint = mSelectHandlePoint - 1;
						mIsHandleSelect = true;
					}
					if (InterpMode == Interpolation::Linear)
					{
						mSelectPointID = id;
						mIsSelect = true;
						mInsertBezierOpenPoint = false;
						mHandleParent = id;
						mSelectHandlePoint = -1;
						mConnectHandlePoint = -1;
						mIsHandleSelect = false;
					}

					mMultiSelectID.clear();
					mMultiSelectID.push_back(id);
				}
				else
				{ 
					addPointtoEnd(); 
				}
				
			}

		}
		
		else if(event->button() == Qt::RightButton)
		{

			if(mSelectPointID >=0)
			{
				deletePoint();
			}

		}


		this->update();
	}
	
	int QCurveLabel::addPointtoEnd()
	{	
		if (!mInsertAtBegin)
		{
			mCoordArray.push_back(mPressCoord);
			buildCoordToResortMap();
			insertHandlePoint(mCoordArray.size() - 1,mPressCoord);

			if (InterpMode == Interpolation::Bezier)
			{
				mInsertBezierOpenPoint = true;
				mSelectPointID = -1;
				mIsSelect = false;
				mHandleParent = mCoordArray.size() - 1;
				mSelectHandlePoint = mHandleParent * 2 + 1;
				mConnectHandlePoint = mSelectHandlePoint - 1;
				mIsHandleSelect = true;
			}
			if (InterpMode == Interpolation::Linear)
			{
				mInsertBezierOpenPoint = false;
				mSelectPointID = mCoordArray.size() - 1;
				mIsSelect = true;
				mHandleParent = mSelectPointID;
				mSelectHandlePoint = -1;
				mConnectHandlePoint = -1;
				mIsHandleSelect = false;
				mInitPosition.set(mCoordArray[mSelectPointID]);
			}

			mMultiSelectID.clear();
			mMultiSelectID.push_back(mCoordArray.size() - 1);
		}
		else if(mInsertAtBegin)
		{
			mCoordArray.insert(mCoordArray.begin(), mPressCoord);
			buildCoordToResortMap();
			insertHandlePoint(0, mPressCoord);


			if (InterpMode == Interpolation::Bezier)
			{
				mInsertBezierOpenPoint = true;
				mSelectPointID = -1;
				mIsSelect = false;
				mHandleParent = 0;
				mSelectHandlePoint = 1;
				mConnectHandlePoint = mSelectHandlePoint - 1;
				mIsHandleSelect = true;
			}
			if (InterpMode == Interpolation::Linear)
			{
				mInsertBezierOpenPoint = false;
				mSelectPointID = 0;
				mIsSelect = true;
				mHandleParent = mSelectPointID;
				mSelectHandlePoint = -1;
				mConnectHandlePoint = -1;
				mIsHandleSelect = false;
				mInitPosition.set(mCoordArray[mSelectPointID]);
			}
		}

		
		return mHandleParent;

	}



	void QCurveLabel::mouseMoveEvent(QMouseEvent* event)
	{
		this->grabKeyboard();
		//constrained
		if (mIsSelect) 
		{
			mCoordArray[mSelectPointID].x = dyno::clamp(event->pos().x(), minX, maxX);
			mCoordArray[mSelectPointID].y = dyno::clamp(event->pos().y(), minY, maxY);

			mDtPosition.set(mCoordArray[mSelectPointID].x-mInitPosition.x , mCoordArray[mSelectPointID].y - mInitPosition.y);

			mHandlePoints[mSelectPointID * 2].x = dyno::clamp(mHandlePoints[mSelectPointID * 2].x + mDtPosition.x, minX, maxX);
			mHandlePoints[mSelectPointID * 2].y = dyno::clamp(mHandlePoints[mSelectPointID * 2].y + mDtPosition.y, minY, maxY);

			mHandlePoints[mSelectPointID * 2 + 1].x = dyno::clamp(mHandlePoints[mSelectPointID * 2 + 1].x + mDtPosition.x, minX, maxX);
			mHandlePoints[mSelectPointID * 2 + 1].y = dyno::clamp(mHandlePoints[mSelectPointID * 2 + 1].y + mDtPosition.y, minY, maxY);

			mInitPosition = mCoordArray[mSelectPointID];

			if (mMultiSelectID.size() > 1)
			{
				for (size_t i = 0; i < mMultiSelectID.size(); i++)
				{
					int tempSelect = mMultiSelectID[i];
					if (tempSelect != mSelectPointID) 
					{
			

						mCoordArray[tempSelect].x = dyno::clamp(mCoordArray[tempSelect].x + mDtPosition.x, minX, maxX);
						mCoordArray[tempSelect].y = dyno::clamp(mCoordArray[tempSelect].y + mDtPosition.y, minY, maxY);

						//
						mHandlePoints[tempSelect * 2].x = dyno::clamp(mHandlePoints[tempSelect * 2].x + mDtPosition.x, minX, maxX);
						mHandlePoints[tempSelect * 2].y = dyno::clamp(mHandlePoints[tempSelect * 2].y + mDtPosition.y, minY, maxY);

						mHandlePoints[tempSelect * 2 + 1].x = dyno::clamp(mHandlePoints[tempSelect * 2 + 1].x + mDtPosition.x, minX, maxX);
						mHandlePoints[tempSelect * 2 + 1].y = dyno::clamp(mHandlePoints[tempSelect * 2 + 1].y + mDtPosition.y, minY, maxY);
					}
				}

			}





			update();
		}
		//constrained Handle
		if (mIsHandleSelect)
		{

			mHandlePoints[mSelectHandlePoint].x = dyno::clamp(event->pos().x(), minX, maxX);
			mHandlePoints[mSelectHandlePoint].y = dyno::clamp(event->pos().y(), minY, maxY);
			
			Vec2f V1 = Vec2f(mHandlePoints[mSelectHandlePoint].x - mCoordArray[mHandleParent].x, mHandlePoints[mSelectHandlePoint].y - mCoordArray[mHandleParent].y);
			Vec2f V2 = V1;
			Vec2f V3 = Vec2f(mHandlePoints[mConnectHandlePoint].x - mCoordArray[mHandleParent].x, mHandlePoints[mConnectHandlePoint].y - mCoordArray[mHandleParent].y);
			V3.normalize();
			Vec2f P = Vec2f(mCoordArray[mHandleParent].x, mCoordArray[mHandleParent].y);
			Vec2f N = -1 * V1.normalize();
			float angle = acos(V3.dot(N))/ M_PI *180;

			if (!mAltKey && mHandleConnection) //!altKey
			{
				if (mInsertBezierOpenPoint)
				{
					mHandlePoints[mConnectHandlePoint] = N * (V2.norm()) + P;
				}
				else 
				{
					mHandlePoints[mConnectHandlePoint] = N * mConnectLength + P;
					mHandlePoints[mConnectHandlePoint].x = dyno::clamp(mHandlePoints[mConnectHandlePoint].x, minX, maxX);
					mHandlePoints[mConnectHandlePoint].y = dyno::clamp(mHandlePoints[mConnectHandlePoint].y, minY, maxY);
				}
			}


			update();
		}


		if (mIsHover == true)
		{
			int tempHover = sqrt(std::pow((event->pos().x() - mCoordArray[mHoverPoint].x), 2) + std::pow((event->pos().y() - mCoordArray[mHoverPoint].y), 2));
			if (tempHover >= selectDistance)
			{
				mHoverPoint = -1;
				mIsHover = false;
			}
		}
		else 
		{
			for (size_t i = 0; i < mCoordArray.size(); i++)
			{
				int temp = sqrt(std::pow((event->pos().x() - mCoordArray[i].x), 2) + std::pow((event->pos().y() - mCoordArray[i].y), 2));

				if (temp < selectDistance)
				{
					mHoverPoint = i;
					mIsHover = true;
					break;
				}
			}
			update();
		}

	}

	void QCurveLabel::updateDataToField()
	{
		Curve s;
		if (mField == nullptr) { return; }
		else
		{
			if (!mReSortCoordArray.empty()) { updateFloatCoordArray(mReSortCoordArray, mFloatCoord); }
			if (!mHandlePoints.empty()) { updateFloatCoordArray(mHandlePoints, mHandleFloatCoord); }

			 CoordtoField(s); 
		}
		mField->setValue(s);
	}

	void QCurveLabel::changeValue(int s)
	{
		this->mMode = (Dir)s;
		Curve ras;

		initializeLine(mMode);

		if (mField == nullptr) { return; }
		else
		{
			updateDataToField();
			if (!mFloatCoord.empty()) { CoordtoField(ras); }
		}

		this->mForceUpdate = true;
		update();

	}

	void QCurveLabel::changeInterpValue(int s)
	{
		this->InterpMode = (Interpolation)s;
		if (InterpMode == Interpolation::Linear) { useBezier = 0; }
		else if (InterpMode == Interpolation::Bezier) { useBezier = 1; }
		this->mForceUpdate = true;
		update();
	}




	void QCurveLabel::CoordtoField(Curve& s)
	{
		s.clearMyCoord();
		for (auto it : mFloatCoord)
		{
			s.addFloatItemToCoord(it.x, it.y, s.mCoord);
		}

		for (auto it : mHandleFloatCoord)
		{
			s.addFloatItemToCoord(it.x, it.y, s.myHandlePoint);
		}

		for (auto it : mCoordArray)
		{
			s.addItemOriginalCoord(it.x, it.y);
		}
		for (auto it : mHandlePoints)
		{
			s.addItemHandlePoint(it.x, it.y);
		}

		s.useBezierInterpolation = useBezier;

		s.resample = LineResample;
		s.useSquard = isSquard;
		s.Spacing = spacing;

		s.NminX = NminX;
		s.NmaxX = NmaxX;
		s.mNewMinY = mNewMinY;
		s.NmaxY = NmaxY;

		s.curveClose = curveClose;

		if (useBezier) { s.mInterpMode = Curve::Interpolation::Bezier; }
		else { s.mInterpMode = Curve::Interpolation::Linear; }

		s.updateBezierCurve();
		s.UpdateFieldFinalCoord();
	}


	void QCurveLabel::copySettingFromField()
	{
		useBezier = mField->getValue().useBezierInterpolation;
		LineResample = mField->getValue().resample;
		spacing = mField->getValue().Spacing;
		curveClose = mField->getValue().curveClose;
		isSquard = mField->getValue().useSquard;
		lockSize = mField->getValue().lockSize;

		if (mField->getValue().mInterpMode == Canvas::Interpolation::Bezier) { InterpMode = Bezier; }
		else { InterpMode = Linear; }

		useSort = false;

	}

}

