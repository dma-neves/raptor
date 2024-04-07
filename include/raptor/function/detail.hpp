//
// Created by david on 05-02-2024.
//

#ifndef RAPTOR_DETAIL_HPP
#define RAPTOR_DETAIL_HPP

#include <type_traits>

#include "function_argument.hpp"

namespace raptor {
    namespace detail {

        template <typename T, typename = void>
        struct is_container : std::false_type {};

        template <typename T>
        struct is_container<T, std::void_t<decltype(std::declval<T>().get_device_data())>> : std::true_type {};

        template <typename T, typename = void>
        struct is_collection : std::false_type {};

        template <typename T>
        struct is_collection<T, std::void_t<decltype(std::declval<T>().size())>> : std::true_type {};

        template <typename T>
        static constexpr bool is_input = std::is_base_of<input, T>::value;

        template <typename T>
        static constexpr bool is_output = std::is_base_of<output, T>::value;
    }
}

#endif //RAPTOR_DETAIL_HPP
