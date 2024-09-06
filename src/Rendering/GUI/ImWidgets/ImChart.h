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
	class ImChart : public ImWidget
	{
		DECLARE_CLASS(ImChart)
	public:
		ImChart();
		~ImChart() override;

		void setCoord(ImVec2 coord);
		ImVec2 getCoord() const;

	public:
		DECLARE_ENUM(InputMode,
			Var = 0,
			Array = 1);

		DEF_VAR_IN(unsigned, FrameNumber, "");
		DEF_VAR_IN(Real, Value, "");
		DEF_ARRAY_IN(Real, Array, DeviceType::GPU, "");

		DEF_ENUM(InputMode, InputMode, InputMode::Array, "");

		DEF_VAR(Real, Min, Real(-1), "");
		DEF_VAR(Real, Max, Real(1), "");
		DEF_VAR(bool, FixHeight, false, "");
		//DEF_VAR(bool, FixCount, false, "");
		DEF_VAR(int, Count, -1, "");
		DEF_VAR(std::string, Title, "chart", "");
		DEF_VAR(bool, OutputFile, false, "");
		DEF_VAR(std::string, OutputPath, "D:/outputTxt", "OutputPath");


	protected:
		virtual void paint();
		virtual void update();
		virtual bool initialize();

		void resetCanvas();
		void varChange();
		void drawArray();
		void drawValue();
		void outputTxt();


	private:

		ImVec2              		mCoord = ImVec2(0, 0);
#ifdef CUDA_BACKEND
		Reduction<Real> 			m_reduce_real;
#endif // CUDA_BACKEND

#ifdef VK_BACKEND
		VkReduce<Real>				m_reduce_real;
#endif // VK_BACKEND


		float min = -1;
		float max = 1;

		Real* valuePtr = nullptr;

		CArray<Real> c_Value;
		std::vector<Real>* valueVec;
		DArray<Real> d_Value;

		std::shared_ptr<Node> mNode;

		int frameNumber = 0;
		int count = 20;

		std::string file_postfix = ".txt";

		bool isOut = false;

	};
};
