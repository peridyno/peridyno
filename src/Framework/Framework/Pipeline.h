#pragma once
#include "Framework/Module.h"

namespace dyno
{

	class ModuleIterator
	{
	public:
		ModuleIterator();

		~ModuleIterator();

		ModuleIterator(const ModuleIterator &iterator);

		ModuleIterator& operator= (const ModuleIterator &iterator);

		bool operator== (const ModuleIterator &iterator) const;

		bool operator!= (const ModuleIterator &iterator) const;

		ModuleIterator& operator++ ();
		ModuleIterator& operator++ (int);

		std::shared_ptr<Module> operator *();

		Module* operator->();

		Module* get();

	protected:

		std::weak_ptr<Module> module;

		friend class Pipeline;
	};

	class Pipeline : public Module
	{
		DECLARE_CLASS(Pipeline)
	public:
		typedef ModuleIterator Iterator;

		Pipeline();
		virtual ~Pipeline();

		Iterator entry();
		Iterator finished();

		unsigned int size();

		void push_back(std::weak_ptr<Module> m);

	private:
		std::weak_ptr<Module> start_module;
		std::weak_ptr<Module> current_module;
		std::weak_ptr<Module> end_module;

		unsigned int num = 0;

		std::map<std::string, Module*> names;
	};
}

