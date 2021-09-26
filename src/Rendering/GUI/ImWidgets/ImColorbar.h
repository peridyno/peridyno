/**
 * Copyright 2017-2021 Xukun Luo
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

#include "ImWidget.h"

namespace dyno
{
	class ImColorbar : public ImWidget
	{
		DECLARE_CLASS(ImColorbar)
	public:
		DECLARE_ENUM(ColorTable,
			Jet = 0,
			Heat = 1);
			
		ImColorbar();
		~ImColorbar() override;

		void setCoord(ImVec2 coord);
		ImVec2 getCoord() const;

	public:
		DEF_ENUM(ColorTable, Type, ColorTable::Jet, "");
		DEF_VAR(Real, Min, Real(0), "");
		DEF_VAR(Real, Max, Real(1), "");

		DEF_ARRAY_IN(Real, Scalar, DeviceType::GPU, "");
		// DEF_ARRAY_IN(Vec3f, Color, DeviceType::CPU, "");
		// DEF_ARRAY_IN(int, Value, DeviceType::GPU, "");

	protected:
		virtual void paint();
		virtual void update();
		virtual bool initialize();

	private:

		ImU32						mMinCol = 0;
		ImU32						mMaxCol = 0;
		int							mNum = 6 + 1;
        ImVec2              		mCoord = ImVec2(0,0);
		DArray<Vec3f> 				mColorBuffer;
		float*						mVal = nullptr;
		ImU32*						mCol = nullptr;
		Reduction<Real> 			m_reduce_real;
		const char*					mTitle;
		float val[6 + 1];
		ImU32 col[6 + 1];

		Vec3f mBaseColor;
		std::shared_ptr<Node> mNode;
	};
};
