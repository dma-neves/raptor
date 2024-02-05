//
// Created by david on 05-02-2024.
//

#ifndef LMARROW_FUNCTION_ARGUMENT_HPP
#define LMARROW_FUNCTION_ARGUMENT_HPP

namespace lmarrow {

    struct input {};

    struct output {};

    template <typename Arg>
    struct in : input {};

    template <typename Arg>
    struct out : output {};

    template <typename Arg>
    struct inout : input, output {};
}

#endif //LMARROW_FUNCTION_ARGUMENT_HPP
