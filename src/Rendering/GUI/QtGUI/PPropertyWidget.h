/**
 * Program:   Qt Property Item
 * Module:    PPropertyWidget.h
 *
 * Copyright 2017-2022 Xiaowei He
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
#include <map>
#include <QWidget>
#include <typeinfo>
#include <memory>


class QVBoxLayout;
class QScrollArea;
class QGridLayout;
class LockerButton;

namespace Qt {
	class QtNode;
}

namespace dyno
{
	class OBase;
	class Node;
	class FBase;
	class Module;

	class PPropertyWidget : public QWidget
	{
		Q_OBJECT
	public:
		explicit PPropertyWidget(QWidget *parent = nullptr);
		~PPropertyWidget();

		virtual QSize sizeHint() const;

//		void clear();

	//signals:
		QWidget* addWidget(QWidget* widget);
		void removeAllWidgets();

	signals:
		void nodeUpdated(std::shared_ptr<Node> node);
		void moduleUpdated(std::shared_ptr<Module> node);

		void stateFieldUpdated(FBase* field, int status);

	public slots:
		void showModuleProperty(std::shared_ptr<Module> module);
		void showNodeProperty(std::shared_ptr<Node> node);

		void showProperty(Qt::QtNode& block);

		//A slot to receive a message when any field widget is updated
		void contentUpdated();

	public:
		struct FieldWidgetMeta {
			using constructor_t = QWidget* (*)(FBase*);
			const std::type_info* type;
			constructor_t constructor;
		};
		static int registerWidget(const FieldWidgetMeta&);
		static FieldWidgetMeta* getRegistedWidget(const std::string&);

		static QWidget* createFieldWidget(FBase* field);


		static std::map<std::string, FieldWidgetMeta> tempGetMeta() { return sFieldWidgetMeta; };
	private:
		static std::map<std::string, FieldWidgetMeta> sFieldWidgetMeta;

		void addScalarFieldWidget(FBase* field, QGridLayout* layout,int j);
		void addArrayFieldWidget(FBase* field);

		void addStateFieldWidget(FBase* field);

		QVBoxLayout* mMainLayout;
		QScrollArea* mScrollArea;
		QGridLayout* mScrollLayout;

		std::vector<QWidget*> mPropertyItems;
		
		LockerButton* mPropertyLabel[3];
		QWidget* mPropertyWidget[3];
		QGridLayout* mPropertyLayout[3];
		bool mFlag[3];

		std::shared_ptr<OBase> mSeleted = nullptr;
	};
}

#define DECLARE_FIELD_WIDGET \
	static int reg_field_widget; \
	static QWidget* createWidget(dyno::FBase*);

#define IMPL_FIELD_WIDGET(_data_type_, _type_) \
	int _type_::reg_field_widget = \
		dyno::PPropertyWidget::registerWidget(dyno::PPropertyWidget::FieldWidgetMeta {&typeid(_data_type_), &_type_::createWidget}); \
	QWidget* _type_::createWidget(dyno::FBase* f) { return new _type_(f); }
