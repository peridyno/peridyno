/*
 * @file point_render_task.h
 * @Basic render task of point
 * @author Wei Chen, Xiaowei He
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public
 * License v2.0. If a copy of the GPL was not distributed with this file, you
 * can obtain one at: http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#pragma once

#include <memory>
#include <vector>

#include <Array/Array.h>
#include "CudaVBOMapper.h"
#include "ShaderProgram.h"

namespace dyno {

class PointRender {
public:
	explicit PointRender();
	~PointRender();

	// disable copy
	PointRender(const PointRender &) = delete;
	PointRender &operator=(const PointRender &) = delete;

	void resize(unsigned int num);

	void setVertexArray(DArray<float3> &pos);
	void setVertexArray(CArray<float3> &pos);

	void setColorArray(DArray<float3> &color);
	void setColorArray(CArray<float3> &color);

	void setPointSize(float point_size);
	void setInstanceSize(float r);
	float pointSize() const;

	void setColor(glm::vec3 color);
	void setColor(DArray<glm::vec3> color);

	void setPointScaleForPointSprite(float point_scale);
	float pointScaleForPointSprite() const;

	void enableUsePointSprite();
	void disableUsePointSprite();
	bool isUsePointSprite() const;

	void renderInstancedSphere();
	void renderSprite();
	void renderPoints();

	int numOfPoints() {
		return m_vertVBO.getSize();
	}

private:
	bool use_point_sprite_ = true;

	float m_instance_size = 0.0025f;

	float point_size_ = 1.0f;
	float point_scale_ = 5.0f; // for point sprite

	unsigned int quadVAO, quadVBO;
	unsigned int instanceVBO;

	CudaVBOMapper<glm::vec3> m_vertVBO;
	CudaVBOMapper<glm::vec3> m_normVBO;
	CudaVBOMapper<glm::vec3> m_vertexColor;

	ShaderProgram m_glsl;
	ShaderProgram m_instancedShader;
};

} // namespace dyno
