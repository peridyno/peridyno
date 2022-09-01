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
#include <stdlib.h>
#include <sstream>
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

		uint size() override { return 1; }

		void setValue(T val);
		T getValue();

		std::string serialize() override { return "Unknown"; }
		bool deserialize(const std::string& str) override { return false; }

		bool isEmpty() override {
			return this->getDataPtr() == nullptr;
		}
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

		this->tick();
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

		inline uint size() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint num);
		void reset();	
		
		void assign(const T& val);
		void assign(std::vector<T>& vals);
		void assign(DArray<T>& vals);

		bool isEmpty() override {
			return this->size() == 0;
		}
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
	void FArray<T, deviceType>::resize(uint num)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr) {
			data = std::make_shared<Array<T, deviceType>>();
		}
		
		data->resize(num);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void dyno::FArray<T, deviceType>::assign(const T& val)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(val);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(std::vector<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(DArray<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::reset()
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->reset();

		this->tick();
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

		inline uint size() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint nx, uint ny);

		void reset();

		void assign(CArray2D<T>& vals);
		void assign(DArray2D<T>& vals);

		bool isEmpty() override {
			return this->getDataPtr() == nullptr;
		}
	};

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::assign(DArray2D<T>& vals)
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}
	
		data->assign(vals);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::assign(CArray2D<T>& vals)
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->assign(vals);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::reset()
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->reset();

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::resize(uint nx, uint ny)
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr) {
			data = std::make_shared<Array2D<T, deviceType>>();
		}
		
		data->resize(ny, ny);

		this->tick();
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

		inline uint size() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(const uint nx, const uint ny, const uint nz);

		void reset();

		void assign(CArray3D<T>& vals);
		void assign(DArray3D<T>& vals);

		bool isEmpty() override {
			return this->getDataPtr() == nullptr;
		}
	};

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::assign(DArray3D<T>& vals)
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->assign(vals);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::assign(CArray3D<T>& vals)
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->assign(vals);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::reset()
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->reset();

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::resize(const uint nx, const uint ny, const uint nz)
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();
		
		data->resize(nx, ny, nz);

		this->tick();
	}

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

		inline uint size() override {
			auto ref = this->getDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint num);

		void resize(const Array<T, deviceType>& arr);

		void assign(const ArrayList<T, DeviceType::CPU>& src);
		void assign(const ArrayList<T, DeviceType::GPU>& src);

		bool isEmpty() override {
			return this->getDataPtr() == nullptr;
		}
	};

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::assign(const ArrayList<T, DeviceType::CPU>& src)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->assign(src);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::assign(const ArrayList<T, DeviceType::GPU>& src)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->assign(src);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::resize(uint num)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->resize(num);

		this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::resize(const Array<T, deviceType>& arr)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->resize(arr);

		this->tick();
	}
}

#include "FSerialization.inl"
