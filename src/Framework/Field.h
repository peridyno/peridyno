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
#include <iostream>
#include "FBase.h"
#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
#include "Array/ArrayList.h"

namespace dyno {
	/*!
	*	\class	Variable
	*	\brief	Variables of build-in data types.
	*/
	template<typename T>
	class FVar : public FBase
	{
	public:
		typedef T				VarType;
		typedef T				DataType;
		typedef FVar<T>		FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FVar);

		FVar(T value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent);
		~FVar() override;

		uint getElementCount() override { return 1; }

		void setValue(T val);
		T getValue();
	};

	template<typename T>
	FVar<T>::FVar(T value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
		: FBase(name, description, fieldType, parent)
	{
		this->setValue(value);
	}

	template<typename T>
	FVar<T>::~FVar()
	{
	};

	template<typename T>
	void FVar<T>::setValue(T val)
	{
		std::shared_ptr<T>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<T>(val);
		}
		else
		{
			*data = val;
		}

		this->update();
	}


	template<typename T>
	T FVar<T>::getValue()
	{
		std::shared_ptr<T>& data = this->getDataPtr();

		return *data;
	}


	template<typename T>
	using HostVarField = FVar<T>;

	template<typename T>
	using DeviceVarField = FVar<T>;

	/**
	 * Define field for Array
	 */
	template<typename T, DeviceType deviceType>
	class FArray : public FBase
	{
	public:
		typedef T							VarType;
		typedef Array<T, deviceType>		DataType;
		typedef FArray<T, deviceType>	FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray);

		~FArray() override;

		inline uint getElementCount() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void setElementCount(uint num);

		void setValue(std::vector<T>& vals);
		void setValue(DArray<T>& vals);
	};

	template<typename T, DeviceType deviceType>
	FArray<T, deviceType>::~FArray()
	{
		if (m_data.use_count() == 1)
		{
			m_data->clear();
		}
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::setElementCount(uint num)
	{
		FBase* topField = this->getTopField();
		FArray<T, deviceType>* derived = dynamic_cast<FArray<T, deviceType>*>(topField);

		if (derived->m_data == nullptr)
		{
			derived->m_data = std::make_shared<Array<T, deviceType>>(num);
		}
		else
		{
			derived->m_data->resize(num);
		}
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::setValue(std::vector<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::setValue(DArray<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);
	}

	template<typename T>
	using HostArrayField = FArray<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArrayField = FArray<T, DeviceType::GPU>;


	/**
	 * Define field for Array2D
	 */
	template<typename T, DeviceType deviceType>
	class FArray2D : public FBase
	{
	public:
		typedef T								VarType;
		typedef Array2D<T, deviceType>			DataType;
		typedef FArray2D<T, deviceType>			FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray2D);

		inline uint getElementCount() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint nx, uint ny);
	};

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::resize(uint nx, uint ny)
	{
		FBase* topField = this->getTopField();
		FArray2D<T, deviceType>* derived = dynamic_cast<FArray2D<T, deviceType>*>(topField);

		if (derived->m_data == nullptr)
		{
			derived->m_data = std::make_shared<Array2D<T, deviceType>>(nx, ny);
		}
		else
		{
			derived->m_data->resize(ny, ny);
		}
	}


	/**
	 * Define field for Array3D
	 */
	template<typename T, DeviceType deviceType>
	class FArray3D : public FBase
	{
	public:
		typedef T								VarType;
		typedef Array3D<T, deviceType>			DataType;
		typedef FArray3D<T, deviceType>			FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray3D);
	};

	/**
	 * Define field for Array
	 */
	template<typename T, DeviceType deviceType>
	class FArrayList : public FBase
	{
	public:
		typedef T								VarType;
		typedef ArrayList<T, deviceType>		DataType;
		typedef FArrayList<T, deviceType>	FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArrayList);

		inline uint getElementCount() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}
	};
}