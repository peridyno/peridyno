#include "QRampWidget.h"

#include <QComboBox>
#include <QGridLayout>
#include <QPainter>

#include "Field.h"

namespace dyno
{
	IMPL_FIELD_WIDGET(Ramp, QRampWidget)

	QRampWidget::QRampWidget(FBase* field)
		: QFieldWidget(field)
	{
		FVar<Ramp>* f = TypeInfo::cast<FVar<Ramp>>(field);
		if (f == nullptr)
		{
			printf("QRamp Nullptr\n");
			return;

		}

		//构建枚举列表
		int curIndex = int(f->getValue().mode);
		int enumNum = f->getValue().count;

		QComboBox* combox = new QComboBox;
		combox->setMaximumWidth(256);
		for (size_t i = 0; i < enumNum; i++)
		{
			auto enumName = f->getValue().DirectionStrings[i];

			combox->addItem(QString::fromStdString(enumName));
		}

		combox->setCurrentIndex(curIndex);

		QGridLayout* layout = new QGridLayout;
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);
		this->setLayout(layout);

		QLabel* name = new QLabel();
		
		name->setFixedSize(80, 18);

		name->setText(FormatFieldWidgetName(field->getObjectName()));
		name->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);//


		QPushButton* unfold = new QPushButton("Squard");

		QDrawLabel* DrawLabel = new QDrawLabel();
		DrawLabel->setMode(combox->currentIndex());
		DrawLabel->setBorderMode(int(f->getValue().Bordermode));
		DrawLabel->setField(f);
		DrawLabel->copyFromField(f->getValue().Originalcoord);
		
		connect(combox, SIGNAL(currentIndexChanged(int)), DrawLabel, SLOT(changeValue(int)));


		QPushButton* button = new QPushButton("Button");
	
		button->setFixedSize(60, 24);
		button->setMaximumWidth(100);

		layout->addWidget(name, 0, 0, Qt::AlignLeft);
		layout->addWidget(DrawLabel,0,1, Qt::AlignCenter);
		layout->addWidget(button,0,2, Qt::AlignRight);
		layout->addWidget(combox, 1, 1, Qt::AlignLeft);
		layout->addWidget(unfold,1,2,Qt::AlignLeft);

		layout->setColumnStretch(0, 0);
		layout->setColumnStretch(1, 5);
		layout->setColumnStretch(2, 0);
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
		printf("paintEvent\n");
		radius = 4;
		int w = this->width();
		int h = this->height();
		minX = 0 + 1.5 * radius;
		maxX = w - 2 * radius;
		minY = 0 + 2 * radius;
		maxY = h - 1.5 * radius;

		if (CoordArray.empty())
		{
			MyCoord FirstCoord;
			MyCoord LastCoord;
			CoordArray.push_back(FirstCoord);
			CoordArray.push_back(LastCoord);
			
			if (Mode == x)
			{
				CoordArray[0].x = minX;
				CoordArray[0].y = (maxY + minY) / 2 ;
				CoordArray[1].x = maxX;
				CoordArray[1].y = (maxY + minY) / 2 ;
			}
			if (Mode == y)
			{
				CoordArray[0].x = (maxX + minX) / 2;
				CoordArray[0].y = minY;
				CoordArray[1].x = (maxX + minX) / 2;
				CoordArray[1].y = maxY;
			}
		}
		

		QPainter painter(this);
		painter.setRenderHint(QPainter::Antialiasing, true);
		//BG
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
		reSortCoordArray.assign(CoordArray.begin(),CoordArray.end());
		reSort(reSortCoordArray);
		for (size_t i = 0; i < reSortCoordArray.size(); i++) 
		{
			QCoordArray.push_back(QPointF(reSortCoordArray[i].x, reSortCoordArray[i].y));
		}
		//绘制曲线
		QPen LinePen = QPen(QPen(QBrush(Qt::white), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
		painter.setPen(LinePen);
		for (size_t i = 0; i < QCoordArray.size() - 1; i++) 
		{
			painter.drawLine(QCoordArray[i], QCoordArray[i+1]);
		}
		//绘制点
		for (size_t i = 0; i < ptNum; i++)
		{
			painter.setBrush(QBrush(Qt::gray, Qt::SolidPattern));
			painter.drawEllipse(CoordArray[i].x - radius, CoordArray[i].y - radius, 2*radius, 2 * radius);
			painter.setPen(QPen(QBrush(QColor(200,200,200), Qt::SolidPattern), 2, Qt::SolidLine, Qt::SquareCap, Qt::BevelJoin));
			painter.drawEllipse(CoordArray[i].x - radius, CoordArray[i].y - radius, 2 * radius, 2 * radius);
			printf("第%d个点 : %d - %d\n",int(i), CoordArray[i].x - radius, CoordArray[i].y - radius);
			
		}
		//Paint SelectPoint
		if (selectPoint != -1) 
		{
			painter.setBrush(QBrush(QColor(80,179,255), Qt::SolidPattern));
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
		printf("绘制时CArray大小：%d\n",CoordArray.size());
	}

	QDrawLabel::~QDrawLabel() 
	{

	}

	QDrawLabel::QDrawLabel(QWidget* parent)
	{
		this->setFixedSize(470, 100);
		this->setMinimumSize(350, 80);
		this->setMaximumSize(1920, 1920);
		w0 = this->width();
		h0 = this->height();
		this->setStyleSheet("background:rgba(110,115,100,1)");
		this->setMouseTracking(true);

	}
	void QDrawLabel::reSort(std::vector<MyCoord>& vector1)
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

	void QDrawLabel::mousePressEvent(QMouseEvent* event) 
	{
		MyCoord pressCoord;
		pressCoord.x = event->pos().x();
		pressCoord.y = event->pos().y();

		for (size_t i = 0; i < CoordArray.size();i++) 
		{
			int temp = sqrt(std::pow((pressCoord.x - CoordArray[i].x),2) + std::pow((pressCoord.y - CoordArray[i].y),2));

			if (temp < selectDistance)
			{
				selectPoint = i;
				isSelect = true;
				break;
			}
		}

		if (!isSelect) 
		{
			CoordArray.push_back(pressCoord);
			selectPoint = CoordArray.size() - 1;
			isSelect = true;

		}
		
		this->update();

	}


	void QDrawLabel::mouseMoveEvent(QMouseEvent* event)
	{
		//移动约束 
		if (isSelect) 
		{
			//首位移动约束 
			if (borderMode == BorderMode::Close && selectPoint <= 1 )
			{
				if (Mode == Dir::x)
				{
					CoordArray[selectPoint].y = dyno::clamp(event->pos().y(),minY, maxY);
				}
				else if (Mode == Dir::y) 
				{
					CoordArray[selectPoint].x = dyno::clamp(event->pos().x(),minX, maxX );
				}
			}
			else
			{
				CoordArray[selectPoint].x = dyno::clamp(event->pos().x(), minX, maxX);
				CoordArray[selectPoint].y = dyno::clamp(event->pos().y(), minY, maxY);
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
		printf("xy : %d - %d\n", event->pos().x(), event->pos().y());


	}
	void QDrawLabel::mouseReleaseEvent(QMouseEvent* event)
	{
		selectPoint = -1;
		isSelect = false;
		//更新数据到field 
		if (field == nullptr){return;}
		else
		{
			updateFloatCoordArray();
			CoordtoField(floatCoord,field);
		}

	}

	void QDrawLabel::changeValue(int s) 
	{
		this->Mode = (Dir)s;

		initializeLine(Mode);
	}

	void QDrawLabel::initializeLine(Dir mode) 
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
}

