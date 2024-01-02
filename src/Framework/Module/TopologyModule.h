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
#include "Module.h"

namespace dyno
{
	typedef int PointType;

class TopologyModule : public OBase
{
	DECLARE_CLASS(TopologyModule)

public:
	typedef PointType				Point;
	typedef VectorND<PointType, 2>	Edge;
	typedef Vector<PointType, 3>	Triangle;
	typedef VectorND<PointType, 4>	Quad;
	typedef VectorND<PointType, 4>	Tetrahedron;
	typedef VectorND<PointType, 5>	Pyramid;
	typedef VectorND<PointType, 6>	Pentahedron;
	typedef VectorND<PointType, 8>	Hexahedron;

	typedef VectorND<PointType, 2>	Edg2Tri;
	typedef VectorND<PointType, 3>	Tri2Edg;

	typedef VectorND<PointType, 2>	Edg2Quad;
	typedef VectorND<PointType, 4>	Quad2Edg;

	typedef VectorND<PointType, 2>	Tri2Tet;
	typedef VectorND<PointType, 4>	Tet2Tri;

	typedef VectorND<PointType, 2>	Tri2Quad;

	typedef VectorND<PointType, 2>	Quad2Hex;
	typedef VectorND<PointType, 2>	Edg2Hex;
	

	
public:
	TopologyModule();
	~TopologyModule() override;

	virtual int getDOF() { return 0; }

	inline void tagAsChanged() { m_topologyChanged = true; }
	inline void tagAsUnchanged() { m_topologyChanged = false; }
	inline bool isTopologyChanged() { return m_topologyChanged; }

	//std::string getModuleType() override { return "TopologyModule"; }

	void update();

protected:

	virtual void updateTopology() {};

private:
	bool m_topologyChanged;
};
}
