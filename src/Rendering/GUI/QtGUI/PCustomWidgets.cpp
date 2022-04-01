/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "PCustomWidgets.h"

namespace dyno
{
	QDoubleSlider::QDoubleSlider(QWidget* pParent /*= NULL*/) :
		QSlider(pParent),
		m_Multiplier(10000.0)
	{
		connect(this, SIGNAL(valueChanged(int)), this, SLOT(setValue(int)));

		setSingleStep(1);

		setOrientation(Qt::Horizontal);
		setFocusPolicy(Qt::NoFocus);
	}

	void QDoubleSlider::setValue(int Value)
	{
		emit valueChanged((double)Value / m_Multiplier);
	}

	void QDoubleSlider::setValue(double Value, bool BlockSignals)
	{
		QSlider::blockSignals(BlockSignals);

		QSlider::setValue(Value * m_Multiplier);

		if (!BlockSignals)
			emit valueChanged(Value);

		QSlider::blockSignals(false);
	}

	void QDoubleSlider::setRange(double Min, double Max)
	{
		QSlider::setRange(Min * m_Multiplier, Max * m_Multiplier);

		emit rangeChanged(Min, Max);
	}

	void QDoubleSlider::setMinimum(double Min)
	{
		QSlider::setMinimum(Min * m_Multiplier);

		emit rangeChanged(minimum(), maximum());
	}

	double QDoubleSlider::minimum() const
	{
		return QSlider::minimum() / m_Multiplier;
	}

	void QDoubleSlider::setMaximum(double Max)
	{
		QSlider::setMaximum(Max * m_Multiplier);

		emit rangeChanged(minimum(), maximum());
	}

	double QDoubleSlider::maximum() const
	{
		return QSlider::maximum() / m_Multiplier;
	}

	double QDoubleSlider::value() const
	{
		int Value = QSlider::value();
		return (double)Value / m_Multiplier;
	}

	QSize QDoubleSpinner::sizeHint() const
	{
		return QSize(30, 20);
	}

	QDoubleSpinner::QDoubleSpinner(QWidget* pParent /*= NULL*/) :
		QDoubleSpinBox(pParent)
	{
	}

	void QDoubleSpinner::setValue(double Value, bool BlockSignals)
	{
		blockSignals(BlockSignals);

		QDoubleSpinBox::setValue(Value);

		blockSignals(false);
	}
}

