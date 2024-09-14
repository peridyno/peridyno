 /**
 * Copyright 2017-2023 Xiaowei He
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
#include <functional>
#include <map>
#include <vector>
#include <tuple>


namespace dyno 
{
	template <typename... Args>
	class EventDispatcher {
	public:
		using EventHandler = std::function<void(Args...)>;

		void addEventListener(const std::string& eventName, const EventHandler& handler) {
			eventHandlers[eventName].push_back(handler);
		}

		void removeEventListener(const std::string& eventName, const EventHandler& handler) {
			auto it = eventHandlers.find(eventName);
			if (it != eventHandlers.end()) {
				it->second.erase(std::remove_if(it->second.begin(), it->second.end(), [&handler](const EventHandler& elem) {
					return &elem == &handler;
					}), it->second.end());
			}
		}

		void callDispatcher(const std::string& eventName, Args... eventData) {
			auto it = eventHandlers.find(eventName);
			if (it != eventHandlers.end()) {
				for (const auto& handler : it->second) {
					handler(eventData...);
				}
			}
		}

	private:
		std::map<std::string, std::vector<EventHandler>> eventHandlers;
	};






}
