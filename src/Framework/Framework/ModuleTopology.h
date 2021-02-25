/**
 * Copyright 2017-2021 Xiaowei He
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
#include "Vector.h"
#include "Framework/Module.h"

namespace dyno
{

class TopologyModule : public Module
{
	DECLARE_CLASS(TopologyModule)

public:
	typedef int PointType;
	typedef PointType					Point;
	typedef FixedVector<PointType, 2>	Edge;
	typedef FixedVector<PointType, 3>	Triangle;
	typedef FixedVector<PointType, 4>	Quad;
	typedef FixedVector<PointType, 4>	Tetrahedron;
	typedef FixedVector<PointType, 5>	Pyramid;
	typedef FixedVector<PointType, 6>	Pentahedron;
	typedef FixedVector<PointType, 8>	Hexahedron;

	typedef FixedVector<PointType, 2>	Edg2Tri;
	typedef FixedVector<PointType, 3>	Tri2Edg;
	typedef FixedVector<PointType, 2>	Tri2Tet;
	typedef FixedVector<PointType, 4>	Tet2Tri;
	
public:
	TopologyModule();
	~TopologyModule() override;

	virtual int getDOF() { return 0; }

	virtual bool updateTopology() { return true; }

	inline void tagAsChanged() { m_topologyChanged = true; }
	inline void tagAsUnchanged() { m_topologyChanged = false; }
	inline bool isTopologyChanged() { return m_topologyChanged; }

	std::string getModuleType() override { return "TopologyModule"; }
private:
	bool m_topologyChanged;
};
}
