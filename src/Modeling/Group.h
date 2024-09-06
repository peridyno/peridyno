/**
 * Copyright 2022 Yuzhong Guo
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
#include "Node/ParametricModel.h"
#include "GLSurfaceVisualModule.h"
#include "GLWireframeVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	class Group : public Node
	{
		DECLARE_TCLASS(Group, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		Group();

	public:

		DEF_VAR(std::string, PointId, "", "Primtive ID");

		DEF_VAR(std::string, EdgeId, "", "Primtive ID");

		DEF_VAR(std::string, PrimitiveId, "", "Primtive ID");

		DEF_ARRAY_IN(int, PointId, DeviceType::GPU, "PointId");

		DEF_ARRAY_IN(int, EdgeId, DeviceType::GPU, "EdgeId");

		DEF_ARRAY_IN(int, PrimitiveId, DeviceType::GPU, "primtive id");


	public:

		std::vector<int> selectedPointID;
		std::vector<int> selectedEdgeID;
		std::vector<int> selectedPrimitiveID;

	protected:
		void resetStates() override;

		void varChanged();

	private:

		void substrFromTwoString(std::string& first, std::string& Second, std::string& line, std::string& MyStr, int& index);

		void updateID(std::string currentString, std::vector<int>& idSet) 
		{
			std::vector<std::string> parmList;
			int parmNum = -1;

			parmNum = getparmNum(currentString);

			int tempStart = -1;

			if (currentString.size())
			{
				std::string first = " ";
				std::string second = " ";

				for (size_t i = 0; i < parmNum; i++)
				{
					std::string tempS;
					tempStart++;
					substrFromTwoString(first, second, currentString, tempS, tempStart);

					parmList.push_back(tempS);
				}
			}

			for (auto it : parmList)
			{
				std::string ss = "\-";
				auto s = it.find(ss);
				int tempint1;
				int tempint2;
				if (s == std::string::npos)
				{
					std::stringstream ss1 = std::stringstream(it);
					ss1 >> tempint1;
					idSet.push_back(tempint1);
				}
				else
				{
					std::stringstream ss1 = std::stringstream(it.substr(0, s));
					std::stringstream ss2 = std::stringstream(it.substr(s + 1, it.size() - 1));

					ss1 >> tempint1;
					ss2 >> tempint2;


					int length = tempint2 - tempint1 + 1;

					for (size_t i = 0; i < length; i++)
					{
						idSet.push_back(tempint1);
						tempint1++;
					}
				}
			}

		}

		int getparmNum(std::string currentString)
		{
			std::string findstr = " ";

			int j = findstr.length();
			int parmNum = 0;
			int pos = 0;
			int k = 0;

			while ((k = currentString.find(findstr, pos)) != -1)
			{
				parmNum++;
				pos = k + j;
			}

			if (parmNum == 0)
			{
				printf("********************* The input file is invalid*********************\n");

			}
			else
			{
				printf("string size : %d \n", parmNum);
			}

			return parmNum;
		}
	};



	IMPLEMENT_TCLASS(Group, TDataType);
}