/*
 * @file line_render_task.cpp
 * @Basic render task of line
 * @author Wei Chen
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */
#include <GL/glew.h>
#include "LineRender.h"


namespace dyno{


#define STRINGIFY(A) #A

	const char * line_render_vertex_shader = "#version 330 compatibility \n" STRINGIFY(

	layout(location = 0) in vec3 vert_pos;
	layout(location = 3) in vec3 vert_col;

	out vec3 frag_vert_col;

	void main()
	{
		frag_vert_col = vert_col;
		gl_Position = gl_ModelViewProjectionMatrix * vec4(vert_pos, 1.0);
	}

	);

	const char * line_render_frag_shader = "#version 330 compatibility \n" STRINGIFY(

		in vec3 frag_vert_col;
	out vec4 frag_color;

	void main()
	{
		frag_color = vec4(frag_vert_col, 1.0);
	}

	);

LineRender::LineRender()
{
    m_shader.createFromCStyleString(line_render_vertex_shader, line_render_frag_shader);
}    

LineRender::~LineRender()
{

}

void LineRender::resize(unsigned int num)
{
	m_vertex.resize(2 * num);
	m_vertexColor.resize(2 * num);
}

void LineRender::setLines(DeviceArray<float3>& pos)
{
	cudaMemcpy(m_vertex.cudaMap(), pos.begin(), sizeof(float3) * m_vertex.getSize(), cudaMemcpyDeviceToDevice);
	m_vertex.cudaUnmap();
}

void LineRender::setLines(HostArray<float3>& pos)
{
	cudaMemcpy(m_vertex.cudaMap(), pos.begin(), sizeof(float3) * m_vertex.getSize(), cudaMemcpyHostToDevice);
	m_vertex.cudaUnmap();
}

void LineRender::setColors(HostArray<float3>& color)
{
	cudaMemcpy(m_vertexColor.cudaMap(), color.begin(), sizeof(float3) * m_vertexColor.getSize(), cudaMemcpyHostToDevice);
	m_vertexColor.cudaUnmap();
}


void LineRender::setLineWidth(float line_width)
{
    m_lineWidth = line_width;
}

float LineRender::getLineWidth() const
{
    return m_lineWidth;
}

void LineRender::display()
{
	m_shader.enable();
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glLineWidth(m_lineWidth);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexColor.getVBO());
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	glEnableVertexAttribArray(3);

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertex.getVBO());
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_LINES, 0, m_vertex.getSize());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glDisableVertexAttribArray(3);

    glPopAttrib();
	m_shader.disable();
}

}//end of namespace dyno
