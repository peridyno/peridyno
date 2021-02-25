/*
 * @file line_render_task.h 
 * @Basic render task of line
 * @author Wei Chen, Xiaowei He
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <memory>
#include <vector>
#include "CudaVBOMapper.h"
#include "ShaderProgram.h"
#include <Array/Array.h>

namespace dyno{

class LineRender
{
public:
    explicit LineRender();
    ~LineRender();

    //disable copy
    LineRender(const LineRender &) = delete;
    LineRender & operator = (const LineRender &) = delete;

	void resize(unsigned int num);

	void setLines(DeviceArray<float3>& pos);
	void setLines(HostArray<float3>& pos);

	void setColors(HostArray<float3>& color);

    void setLineWidth(float line_width);
    float getLineWidth() const;

    void display();  //Note: we design renderTaskImpl to be "protected" as the same reason for "TriangleCustomColor::renderTaskImpl".

private:
    float m_lineWidth = 2.0;

	ShaderProgram m_shader;

	CudaVBOMapper<glm::vec3> m_vertex;
	CudaVBOMapper<glm::vec3> m_vertexColor;
};
    
}