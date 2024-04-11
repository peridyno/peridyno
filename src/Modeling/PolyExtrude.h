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
#include "Group.h"

namespace dyno
{
	template<typename TDataType>
	class PolyExtrude : public Group<TDataType>
	{
		DECLARE_TCLASS(PolyExtrude, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;



		PolyExtrude();

	protected:


		struct Prim_point_Point
		{
			int oriP[10];

			int newP[10];

			int primID = -1;

			Prim_point_Point()
			{
				iniArray();
			};

			Prim_point_Point(bool useOri, int prim, int p0, int p1, int p2)
			{
				iniArray();

				if (useOri)
				{
					this->oriP[0] = p0;
					this->oriP[1] = p1;
					this->oriP[2] = p2;
				}
				else
				{
					this->newP[0] = p0;
					this->newP[1] = p1;
					this->newP[2] = p2;
				}
				this->primID = prim;
			};
			Prim_point_Point(bool useOri, int prim, int p)
			{
				iniArray();

				if (useOri)
				{
					for (int i = 0; i < sizeof(oriP) / sizeof(oriP[0]); i++)
					{
						if (oriP[i] == -1) { oriP[i] = p; }
					}
				}
				else
				{
					for (int i = 0; i < sizeof(newP) / sizeof(newP[0]); i++)
					{
						if (newP[i] == -1) { newP[i] = p; }
					}
				}
				this->primID = prim;
			};
			void setNewID_byOriID(int np, int op)
			{
				for (int i = 0; i < sizeof(oriP) / sizeof(oriP[0]); i++)
				{
					if (oriP[i] == op)
					{
						this->newP[i] = np;
						break;
					}

				}
			}
			void iniArray()
			{
				for (int i = 0; i < sizeof(oriP) / sizeof(oriP[0]); i++)
				{
					oriP[i] = -1;
				}

				for (int i = 0; i < sizeof(newP) / sizeof(newP[0]); i++)
				{
					newP[i] = -1;
				}
			}
		};


		struct point_layer
		{
			int layer = -1;
			int oriPoint = -1;

			point_layer() {}
			point_layer(int p, int L)
			{
				this->layer = L;
				this->oriPoint = p;
			};

			bool operator<(const point_layer& other) const
			{
				if (this->layer < other.layer)
				{
					return true;
				}
				else if (this->layer == other.layer)
				{
					if (this->oriPoint < other.oriPoint)
						return true;
					else
						return false;
				}
				else
				{
					return false;
				}
			}

			int getpoint() { return this->oriPoint; }
			int getlayer() { return this->layer; }
		};


	public:

		DEF_VAR(unsigned, Divisions, 1, "Divisions");

		DEF_VAR(Real, Distance, 0.2, "Distance");

		DEF_INSTANCE_IN(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_STATE(TriangleSet<TDataType>, TriangleSet, "");

		DEF_INSTANCE_STATE(EdgeSet<TDataType>, NormalSet, "");

		DEF_VAR(bool, ReverseNormal, false, "ReverseNormal");

	protected:
		void resetStates() override;

		void pointcount(std::map<int, int>& mapp, int p) 
		{
			if (mapp.count(p))
			{
				mapp[p] = mapp.at(p) + 1;
			}
			else
			{
				mapp[p] = 1;
			}
		}

		void substrFromTwoString(std::string& first, std::string& Second, std::string& line, std::string& MyStr, int& index);

	private:
		void varChanged();

		void extrude(std::vector<Coord>& vertices,std::vector<TopologyModule::Triangle>& triangles);

		std::shared_ptr<GLWireframeVisualModule> glModule3;


	};



	IMPLEMENT_TCLASS(PolyExtrude, TDataType);
}