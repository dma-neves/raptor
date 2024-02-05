//
// Created by david on 05-02-2024.
//

#ifndef LMARROW_DETAIL_HPP
#define LMARROW_DETAIL_HPP

#include <type_traits>

namespace lmarrow {
    namespace detail {

        template <typename T, typename = void>
        struct is_container : std::false_type {};

        template <typename T>
        struct is_container<T, std::void_t<decltype(std::declval<T>().get_device_ptr())>> : std::true_type {};


        template <typename T, typename = void>
        struct is_collection : std::false_type {};

        template <typename T>
        struct is_collection<T, std::void_t<decltype(std::declval<T>().size())>> : std::true_type {};
    }
}

#endif //LMARROW_DETAIL_HPP
