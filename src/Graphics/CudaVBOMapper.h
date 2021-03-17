/**
 * Copyright 2021 Xiaowei He
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
#include "Platform.h"
#include <GL/glew.h>
#include <cuda_gl_interop.h> 

namespace dyno{

template <typename T>
class CudaVBOMapper
{
public:
	CudaVBOMapper()
	{
		m_vbo = 0;
		m_size = 0;
		m_cudaGraphicsResource = NULL;

		glGenBuffers(1, &m_vbo);
	}


	CudaVBOMapper(unsigned int num)
	{
		resize(num);
	}

	CudaVBOMapper(const CudaVBOMapper &) = delete;
	CudaVBOMapper & operator = (const CudaVBOMapper &) = delete;

    ~CudaVBOMapper()
	{
		release();
	}
	
	void resize(unsigned int num)
	{
 		if (m_size != 0)
 		{
			
			cuSafeCall(cudaGraphicsUnregisterResource(m_cudaGraphicsResource));
			//glDeleteBuffers(1, &m_vbo);
			//release();
 		}

		m_size = num;
		
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, m_size * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		cuSafeCall(cudaGraphicsGLRegisterBuffer(&m_cudaGraphicsResource, m_vbo, cudaGraphicsMapFlagsWriteDiscard));
	}

	void release()
	{
		if (m_vbo != 0)
		{
			glDeleteBuffers(1, &m_vbo);
		}
// 		if (m_cudaGraphicsResource != NULL)
// 		{
// 			cuSafeCall(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0));
// 		}
		m_size = 0;
	}

    T* cudaMap()
	{
		cuSafeCall(cudaGraphicsMapResources(1, &m_cudaGraphicsResource, 0));

		T* dataPtr = nullptr;
		size_t byte_size;
		cuSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dataPtr, &byte_size, m_cudaGraphicsResource));

		return dataPtr;
	}

    void cudaUnmap()
	{
		cuSafeCall(cudaGraphicsUnmapResources(1, &m_cudaGraphicsResource, 0));
	}

	unsigned int getVBO()
	{
		return m_vbo;
	}

	unsigned int getSize()
	{
		return m_size;
	}

private:
	int m_size;
	unsigned int m_vbo;
    cudaGraphicsResource * m_cudaGraphicsResource;
};



}//end of namespace dyno
