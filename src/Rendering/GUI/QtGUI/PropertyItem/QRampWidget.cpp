#include "QRampWidget.h"

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
	IMPL_FIELD_WIDGET(Ramp, QRampWidget)

	QRampWidget::QRampWidget(FBase* field)
		: QFieldWidget(field)
	{
		printf("QRampWidget\n");
		FVar<Ramp>* f = TypeInfo::cast<FVar<Ramp>>(field);
		if (f == nullptr)
		{
			printf("QRamp Nullptr\n");
			return;

		}



		//Enum List    InterpMode
		int curIndex2 = int(f->getValue().getInterpMode());
		int enumNum = f->getValue().InterpolationCount;
		QComboBox* combox2 = new QComboBox;
		combox2->setMaximumWidth(256);
		combox2->addItem(QString::fromStdString("Linear"));
		combox2->addItem(QString::fromStdString("Bezier"));
			
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
		QDrawLabel* DrawLabel = new QDrawLabel();
		DrawLabel->useBezier = f->getValue().getInterpMode() == Canvas::Interpolation::Bezier ? true : false;

		DrawLabel->setField(f);

		DrawLabel->copySettingFromField();

		DrawLabel->updateLabelShape(f->getValue().isSquard());

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
		spacingSlider->setValue(f->getValue().getSpacing());
		spacingSlider->id = -1;
		mQDoubleSpinner* spacingSpinner = new mQDoubleSpinner;
		spacingSpinner->setRange(1, 40);
		spacingSpinner->setValue(f->getValue().getSpacing());
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

		QHBoxLayout* Hlayout1 = new QHBoxLayout;
		Hlayout1->setContentsMargins(0, 0, 0, 0);
		Hlayout1->setSpacing(0);


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
		Checkbox->QAbstractButton::setChecked(f->getValue().getResample());

		connect(Checkbox, SIGNAL(mValueChanged(int)), DrawLabel, SLOT(setLinearResample(int)));
		connect(Checkbox, SIGNAL(mValueChanged(int)), spacingSlider, SLOT(setNewVisable(int)));
		connect(Checkbox, SIGNAL(mValueChanged(int)), spacingSpinner, SLOT(setNewVisable(int)));

		if (f->getValue().getResample() == false)
		{
			spacingSlider->setVisible(false);
			spacingSpinner->setVisible(false);
			spacingName->setVisible(false);
		}


		boolLayout->addWidget(boolName, 0);
		boolLayout->addStretch(1);
		boolLayout->addWidget(Checkbox, 0);

		//if (f->getDataPtr()->InterpMode == Ramp::Interpolation::Bezier) { boolName->setVisible(0); Checkbox->setVisible(0); }


		QVBoxLayout* TotalLayout = new QVBoxLayout();
		TotalLayout->setContentsMargins(0, 0, 0, 0);
		TotalLayout->setSpacing(5);
		TotalLayout->addLayout(Gridlayout);
		TotalLayout->addLayout(Hlayout1);
		TotalLayout->addLayout(boolLayout);
		TotalLayout->addLayout(SpacingHlayout);

		this->setLayout(TotalLayout);
		
	}

	void QRampWidget::updateField()
	{
		FVar<Ramp>* f = TypeInfo::cast<FVar<Ramp>>(field());
		if (f == nullptr)
		{
			return;
		}

	}

	void QDrawLabel::paintEvent(QPaintEvent* event)
	{

		//设置半径
		radius = 4;
		int w = this->width();
		int h = this->height();
		
		//设置最大最小坐标
		minX = 0 + 1.5 * radius;
		maxX = w - 2 * radius;
		minY = 0 + 2 * radius;
		maxY = h - 1.5 * radius;

		//如果CoordArray为空，则从field中取数据
		if (mCoordArray.empty())
		{
			if (mField->getValue().getUserPoints().empty())		//如果field中没有Widget传回的数据，则从field本身的Coord进行初始化
			{
				this->copyFromField(mField->getValue().FE_MyCoord, mCoordArray);
				mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());
				reSort(mReSortCoordArray);
				buildCoordToResortMap();
				this->copyFromField(mField->getValue().FE_HandleCoord, mHandlePoints);
			}
			else		//否则直接取field存的qt坐标
			{
				this->copyFromField(mField->getValue().getUserPoints(), mCoordArray);
				//this->copyFromField(field->getDataPtr()->OriginalHandlePoint, HandlePoints);
				this->copyFromField(mField->getValue().getUserHandles(), mHandlePoints);
			}

		}


		if (mCoordArray.empty())			//如果是close模式，且没有初始化数据，则在空画布创建两个点，作为边界点
		{
			if (mMode == x)		//direction为x创建点的逻辑
			{
				if (mGeneratorXmin)
				{
					MyCoord FirstCoord;
					mCoordArray.push_back(FirstCoord);

					mCoordArray[mCoordArray.size() - 1].x = minX;
					mCoordArray[mCoordArray.size() - 1].y = (maxY + minY) / 2;
					std::swap(mCoordArray[0], mCoordArray[mCoordArray.size() - 1]);
					mGeneratorXmin = 0;
				}
				if (mGeneratorXmax)
				{
					MyCoord LastCoord;
					mCoordArray.push_back(LastCoord);

					mCoordArray[mCoordArray.size() - 1].x = maxX;
					mCoordArray[mCoordArray.size() - 1].y = (maxY + minY) / 2;
					std::swap(mCoordArray[1], mCoordArray[mCoordArray.size() - 1]);
					mGeneratorXmax = 0;
				}
			}
		}

		mReSortCoordArray.assign(mCoordArray.begin(), mCoordArray.end());		//用CoordArray为reSortCoordArray赋值
		reSort(mReSortCoordArray);		//对reSortCoordArray重排序


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

		QVector<QPointF> QCoordArray;//创建QPointF以绘制点
		for (size_t i = 0; i < mReSortCoordArray.size(); i++)
		{
			QCoordArray.push_back(QPointF(mReSortCoordArray[i].x, mReSortCoordArray[i].y));
		}


		buildCoordToResortMap();	//构建map用以通过CoordArray查找排序后reSortCoordArray的元素id

		if (mHandlePoints.empty())	//通过CoordArray构建贝塞尔控制柄的点HandlePoints
		{
			buildHandlePointSet();
		}

		//绘制曲线
		QPen LinePen = QPen(QPen(QBrush(QColor(200,200,200)), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePen);

		mPath.clear();
		if (useBezier)		//画贝塞尔曲线或折线
		{
			//绘制贝塞尔曲线
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

			}
		}

		QPen LinePenWhite = QPen(QPen(QBrush(Qt::white), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePenWhite);

		//绘制点
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

		//绘制控制柄
		if (useBezier) 
		{
			////绘制全部控制柄
			//{
			//	for (size_t i = 0; i < CoordArray.size(); i++)
			//	{
			//		int f = i * 2;
			//		int s = i * 2 + 1;
			//		painter.drawLine(QPointF(CoordArray[i].x, CoordArray[i].y), QPointF(HandlePoints[f].x, HandlePoints[f].y));
			//		painter.drawLine(QPointF(CoordArray[i].x, CoordArray[i].y), QPointF(HandlePoints[s].x, HandlePoints[s].y));
			//	}
			//	for (size_t i = 0; i < HandlePoints.size(); i++)
			//	{
			//		painter.drawEllipse(HandlePoints[i].x - radius, HandlePoints[i].y - radius, 2 * radius, 2 * radius);
			//	}
			//}

			if (mHandleParent != -1)
			{
				int f = mHandleParent * 2;
				int s = mHandleParent * 2 + 1;
				//绘制控制柄
				painter.drawLine(QPointF(mCoordArray[mHandleParent].x, mCoordArray[mHandleParent].y), QPointF(mHandlePoints[f].x, mHandlePoints[f].y));
				painter.drawLine(QPointF(mCoordArray[mHandleParent].x, mCoordArray[mHandleParent].y), QPointF(mHandlePoints[s].x, mHandlePoints[s].y));
				//绘制控制点
				painter.drawEllipse(mHandlePoints[f].x - radius, mHandlePoints[f].y - radius, 2 * radius, 2 * radius);
				painter.drawEllipse(mHandlePoints[s].x - radius, mHandlePoints[s].y - radius, 2 * radius, 2 * radius);
				// 绘制父点
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

		//绘制选中点
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

	QDrawLabel::~QDrawLabel() 
	{

	}


	QDrawLabel::QDrawLabel(QWidget* parent)
	{
		this->setLabelSize(w, h, w, w);
		this->setStyleSheet("background:rgba(110,115,100,1)");
		this->setMouseTracking(true);


		if (mMode == x)
		{
			for (auto it : mCoordArray)
			{
				if (it.x == minX)
				{
					mGeneratorXmin = 0;
				}
				else
				{
					if (!mGeneratorXmin) { mMinIndex++; }
				}
				if (it.x == maxX)
				{
					mGeneratorXmax = 0;
				}
				else
				{
					if (!mGeneratorXmax) { mMaxIndex++; }
				}
			}

			if (!mGeneratorXmin) 
			{
				std::swap(mCoordArray[0], mCoordArray[mMinIndex++]);
			}
			if (!mGeneratorXmax)
			{
				std::swap(mCoordArray[1], mCoordArray[mMaxIndex++]);
			}
		}
		if (mMode == y)
		{
			for (auto it : mCoordArray)
			{
				if (it.y == minY) { mGeneratorYmin = 0; }
				if (it.y == maxY) { mGeneratorYmax = 0; }
			}

			if (!mGeneratorYmin)
			{
				std::swap(mCoordArray[0], mCoordArray[mMinIndex++]);
			}
			if (!mGeneratorYmax)
			{
				std::swap(mCoordArray[1], mCoordArray[mMaxIndex++]);
			}
		}
	}



	void QDrawLabel::mousePressEvent(QMouseEvent* event) 
	{

		//鼠标左键

		mPressCoord.x = event->pos().x();
		mPressCoord.y = event->pos().y();

		if (mShiftKey)	//多选
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
		else if (!mShiftKey && !mAltKey)		//单选
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
		
		//判断是否点击贝塞尔控制柄
		if (useBezier)
		{
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
				
					{//判断Handle是否联动
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
		//未点击任何点则插入点
		if (!mIsSelect && !mIsHandleSelect)
		{
			//close模式下插入点并自动排序
			addPointtoEnd();
		}
		//鼠标右键
		else if(event->button() == Qt::RightButton)
		{
			if (mSelectPointID > 1) 
			{
				deletePoint();
			}
		}

		this->update();
	}

	//尾插
	int QDrawLabel::addPointtoEnd() 
	{

			mCoordArray.push_back(mPressCoord);
			buildCoordToResortMap();
			insertElementToHandlePointSet(mCoordArray.size() - 1);

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



		return mHandleParent;

	}



	void QDrawLabel::mouseMoveEvent(QMouseEvent* event)
	{
		this->grabKeyboard();
		//移动约束 
		if (mIsSelect) 
		{
			//首位移动约束 ——单选

			if (mSelectPointID <= 1)
			{
				if (mMode == Dir::x)
				{
					mCoordArray[mSelectPointID].y = dyno::clamp(event->pos().y(), minY, maxY);
				}
				else if (mMode == Dir::y)
				{
					mCoordArray[mSelectPointID].x = dyno::clamp(event->pos().x(), minX, maxX);
				}
			}
			else
			{
				mCoordArray[mSelectPointID].x = dyno::clamp(event->pos().x(), minX, maxX);
				mCoordArray[mSelectPointID].y = dyno::clamp(event->pos().y(), minY, maxY);
			}

			mDtPosition.set(mCoordArray[mSelectPointID].x-mInitPosition.x , mCoordArray[mSelectPointID].y - mInitPosition.y);

			mHandlePoints[mSelectPointID * 2].x = dyno::clamp(mHandlePoints[mSelectPointID * 2].x + mDtPosition.x, minX, maxX);
			mHandlePoints[mSelectPointID * 2].y = dyno::clamp(mHandlePoints[mSelectPointID * 2].y + mDtPosition.y, minY, maxY);

			mHandlePoints[mSelectPointID * 2 + 1].x = dyno::clamp(mHandlePoints[mSelectPointID * 2 + 1].x + mDtPosition.x, minX, maxX);
			mHandlePoints[mSelectPointID * 2 + 1].y = dyno::clamp(mHandlePoints[mSelectPointID * 2 + 1].y + mDtPosition.y, minY, maxY);

			mInitPosition = mCoordArray[mSelectPointID];

			//多选
			if (mMultiSelectID.size() > 1)
			{
				for (size_t i = 0; i < mMultiSelectID.size(); i++)
				{
					int tempSelect = mMultiSelectID[i];
					if (tempSelect != mSelectPointID) 
					{
						if (tempSelect <= 1)
						{
							if (mMode == Dir::x)
							{
								mCoordArray[tempSelect].y = dyno::clamp(mCoordArray[tempSelect].y + mDtPosition.y, minY, maxY);
							}
							else if (mMode == Dir::y)
							{
								mCoordArray[tempSelect].x = dyno::clamp(mCoordArray[tempSelect].x + mDtPosition.x, minX, maxX);
							}
						}
						else
						{
							mCoordArray[tempSelect].x = dyno::clamp(mCoordArray[tempSelect].x + mDtPosition.x, minX, maxX);
							mCoordArray[tempSelect].y = dyno::clamp(mCoordArray[tempSelect].y + mDtPosition.y, minY, maxY);
						}

						mHandlePoints[tempSelect * 2].x = dyno::clamp(mHandlePoints[tempSelect * 2].x + mDtPosition.x, minX, maxX);
						mHandlePoints[tempSelect * 2].y = dyno::clamp(mHandlePoints[tempSelect * 2].y + mDtPosition.y, minY, maxY);

						mHandlePoints[tempSelect * 2 + 1].x = dyno::clamp(mHandlePoints[tempSelect * 2 + 1].x + mDtPosition.x, minX, maxX);
						mHandlePoints[tempSelect * 2 + 1].y = dyno::clamp(mHandlePoints[tempSelect * 2 + 1].y + mDtPosition.y, minY, maxY);
					}
				}

			}





			update();
		}
		//控制柄移动
		if (mIsHandleSelect)
		{
			//首位移动约束 
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

	void QDrawLabel::updateDataToField() 
	{
		Ramp s;

		//更新数据到field 
		if (mField == nullptr) { return; }
		else
		{
			if (!mReSortCoordArray.empty()) { updateFloatCoordArray(mReSortCoordArray, mFloatCoord); }
			if (!mHandlePoints.empty()) { updateFloatCoordArray(mHandlePoints, mHandleFloatCoord); }

			 CoordtoField(s); 
		}



		mField->setValue(s);

	}

	void QDrawLabel::changeValue(int s) 
	{
		this->mMode = (Dir)s;
		Ramp ras;

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

	void QDrawLabel::changeInterpValue(int s)
	{
		this->InterpMode = (Interpolation)s;
		if (InterpMode == Interpolation::Linear) { useBezier = 0; }
		else if (InterpMode == Interpolation::Bezier) { useBezier = 1; }
		this->mForceUpdate = true;
		update();
	}





	void QDrawLabel::CoordtoField(Ramp& s)
	{
		s.clearMyCoord();
		for (auto it : mFloatCoord)
		{
			s.addFloatItemToCoord(it.x, it.y, s.getUserPoints());
		}

		for (size_t i = 0; i < mReSortCoordArray.size(); i++)
		{
			int sa = mMapResortIDtoOriginalID.find(i)->second;

			int f = 2 * sa;
			int e = 2 * sa + 1;

			s.addFloatItemToCoord(mHandleFloatCoord[f].x, mHandleFloatCoord[f].y, s.getUserHandles());
			s.addFloatItemToCoord(mHandleFloatCoord[e].x, mHandleFloatCoord[e].y, s.getUserHandles());
		}


		s.getResample() = LineResample;
		s.getSpacing() = spacing;

		s.getClose() = curveClose;

		if (useBezier) { s.getInterpMode() = Ramp::Interpolation::Bezier; }
		else { s.getInterpMode() = Ramp::Interpolation::Linear; }

		s.updateBezierCurve();
		s.UpdateFieldFinalCoord();
	}



	void QDrawLabel::copySettingFromField()
	{
		LineResample = mField->getValue().getResample();
		spacing = mField->getValue().getSpacing();
		curveClose = mField->getValue().getClose();

		if (mField->getValue().getInterpMode() == Canvas::Interpolation::Bezier) { InterpMode = Bezier; }
		else { InterpMode = Linear; }

		useSort = true; 

	}

}

