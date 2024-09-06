#pragma once
#include <QString>

#include "FilePath.h"

#if defined(PERIDYNO_QTGUI_EXPORTS)
#define PERIDYNO_QTGUI_API Q_DECL_EXPORT
#else
#define PERIDYNO_QTGUI_API Q_DECL_IMPORT
#endif

namespace dyno {
// 	inline QString qstr(const FilePath& p) {
// 		return QString::fromStdString(p.string());
// 	}
}
