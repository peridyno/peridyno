/**
 * Copyright 2025 Yuzhong Guo
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

#include "ToolBar/TabToolbar.h"

#include <QToolButton>
#include <QPushButton>
#include <QHBoxLayout>
#include <QFrame>

namespace dyno
{
	class PMaterialEditorToolBar : public QFrame
	{
		Q_OBJECT
	public:
		PMaterialEditorToolBar(QWidget* parent = nullptr);

		QAction* addAction(QToolButton::ToolButtonPopupMode type, QAction* action, QMenu* menu = nullptr);

		QAction* updateAction() { return mActionUpdate; }
		QAction* saveAction() { return mActionSave; }
		QAction* reorderAction() { return mActionReorder; }
		QAction* realTimeUpdateAction() { return mActionRealtime; }

		void setRealtime(bool realtime) { mActionRealtime->setChecked(realtime); }

	private:
		QHBoxLayout* mLayout;

		QAction* mActionSave;
		QAction* mActionReorder;
		QAction* mActionUpdate;
		QAction* mActionRealtime;

	};
}
