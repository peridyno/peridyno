/**
 * Program:   Qt Property Item
 * Module:    QRampWidget.h
 *
 * Copyright 2023 Yuzhong Guo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "QFieldWidget.h"
#include "QtGUI/PPropertyWidget.h"

#include "Ramp.h"

namespace dyno
{
	class QRampWidget : public QFieldWidget
	{
		Q_OBJECT
	public:
		DECLARE_FIELD_WIDGET

		QRampWidget(FBase* field);
		~QRampWidget() override {};

	public slots:
		void updateField();

	private:
	};

	class QDrawLabel : public QWidget
	{
		Q_OBJECT
	public:
		struct MyCoord
		{
			int x = 0;
			int y = 0;
			void set(int a, int b) 
			{
				this->x = a;
				this->y = b;
			}
		};
		struct Coord0_1
		{
			float x = 0;
			float y = 0;
			void set(float a, float b) 
			{
				this->x = a;
				this->y = b;
			}
		};

		enum Dir
		{
			x = 0,
			y = 1,
		};

		enum BorderMode
		{
			Open = 0,
			Close = 1,
		};

		enum RectShape
		{
			Rect = 0,
			Squard= 1,

		};

	public:
		QDrawLabel(QWidget* parent = nullptr);
		~QDrawLabel();
		Dir setMode(int s) { return Dir(s); }
		void setBorderMode(int border) { if (border == 0) {borderMode = Open;}else{borderMode = Close;}}
		void setField(FVar<Ramp>* f) { field = f; }
		void copyFromField(std::vector<Ramp::MyCoord2D> coord01) 
		{
			auto ff = field->getValue();
			if (coord01.size()) 
			{
				for (auto it : coord01)
				{
					Coord0_1 s;
					s.set(it.x, it.y);
					reSortCoordArray.push_back(ZeroOneToCoord(s,ff.oMinX, ff.oMaxX, ff.oMinY, ff.oMaxY));
					//printf("RCApush值：%d\n", ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).x, ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).y);
				}
				printf("拷贝的RCA大小%d\n", reSortCoordArray.size());

				CoordArray.push_back(reSortCoordArray[0]);
				CoordArray.push_back(reSortCoordArray[reSortCoordArray.size()-1]);
				for (size_t i = 1;i < reSortCoordArray.size()-1;i++)
				{
					CoordArray.push_back(reSortCoordArray[i]);


				}
			}
			printf("拷贝的CA大小%d\n",CoordArray.size());
			update();
		}
		void copyFromField(std::vector<Ramp::OriginalCoord> coord01)
		{
			auto ff = field->getValue();
			if (coord01.size())
			{
				for (auto it : coord01)
				{
					MyCoord s;
					s.set(it.x, it.y);
					CoordArray.push_back(s);
					//printf("RCApush值：%d\n", ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).x, ZeroOneToCoord(s, ff->oMinX, ff->oMaxX, ff->oMinY, ff->oMaxY).y);
				}
				
			}
			printf("拷贝的CA大小%d\n", CoordArray.size());
			update();
		}

	public slots:
		void changeValue(int s);

	protected:
		void paintEvent(QPaintEvent* event);
		void mouseMoveEvent(QMouseEvent* event) override;
		void mousePressEvent(QMouseEvent* event)override;
		void mouseReleaseEvent(QMouseEvent* event)override;
		void reSort(std::vector<MyCoord>& vector1);
		void initializeLine(Dir mode );
		void updateFloatCoordArray() { floatCoord.clear(); for (auto it : reSortCoordArray) { floatCoord.push_back(CoordTo0_1Value(it)); } }
		void CoordtoField(std::vector<Coord0_1> CA, FVar<Ramp>* f) 
		{	
			auto s = f->getValue();
			s.clearMyCoord();
			for (auto it : CA) 
			{
				s.addItemMyCoord(it.x,it.y);

			}
			for (auto it : CoordArray)
			{
				s.addItemOriginalCoord(it.x, it.y);
			}
			s.setOriginalCoord(minX,maxX,minY,maxY);
		}
		Coord0_1 CoordTo0_1Value(MyCoord& coord) 
		{//曲线坐标转换到0-1浮点值，并反转Y轴
			Coord0_1 s;
			float x = float(coord.x);
			float y = float(coord.y);
			float fmaxX = float(maxX);
			float fminX = float(minX);
			float fmaxY = float(maxY);
			float fminY = float(minY);

			s.x = (x - fminX) / (fmaxX - fminX);
			s.y = 1 - (y - fminY) / (fmaxY - fminY);
			return s;
		}
		MyCoord ZeroOneToCoord(Coord0_1& value, int x0,int x1,int y0, int y1) 
		{//0-1浮点值到曲线坐标
			MyCoord s;
			printf("x0: %d x1: %d y0: %d y1: %d\n", x0, x1, y0, y1);
			s.x = int(value.x * float(x1 - x0)) + x0;
			s.y = int((1-value.y) * float(y1 - y0)) + y0;
			std::cout << value.x << " --- " << value.y << std::endl;
			std::cout << s.x << " *** " << s.y << std::endl;
			return s;
		}

		static bool sortx(MyCoord a, MyCoord b)
		{
			if (a.x < b.x) { return true; }
			else { return false; }
		};
		static bool sorty(MyCoord a, MyCoord b)
		{
			if (a.y < b.y) { return true; }
			else { return false; }
		};
		
		std::vector<MyCoord> CoordArray;
		std::vector<MyCoord> reSortCoordArray;
		MyCoord hoverCoord;
		MyCoord selectCoord;
		std::vector<Coord0_1> floatCoord;

	private:
		int w0;
		int h0;
		int currentW;
		int currentH;
		Dir Mode = x;
		BorderMode borderMode = Close;
		int hoverPoint = -1;
		int selectPoint = -1;
		bool isSelect = false;
		bool isHover = false;
		int maxX = 0;
		int maxY = 0;
		int minX = 0;
		int minY = 0;
		int selectDistance = 10;
		int radius = 4;
		FVar<Ramp>* field;

	};



}
